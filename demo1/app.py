from flask import Flask, render_template, request, redirect, url_for, send_file
import numpy as np
from scipy.io import loadmat, savemat
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
import traceback  # 添加这行导入
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from models.sc_ss_mtr import SC_SS_MTr
from utils.metrics import calculate_metrics
import io

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Global variables to store data between steps
class DataStore:
    hyperspectral_data = None
    label_data = None
    processed_data = None


data_store = DataStore()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'hyperspectral' not in request.files or 'label' not in request.files:
        return 'No files uploaded', 400

    hyperspectral_file = request.files['hyperspectral']
    label_file = request.files['label']

    # Save files temporarily
    hyperspectral_path = os.path.join(app.config['UPLOAD_FOLDER'], 'hyperspectral.mat')
    label_path = os.path.join(app.config['UPLOAD_FOLDER'], 'label.mat')

    # Save and load hyperspectral data
    try:
        hyperspectral_file.save(hyperspectral_path)
        mat_contents = loadmat(hyperspectral_path)
        var_names = [name for name in mat_contents.keys() if not name.startswith('__')]
        if len(var_names) == 0:
            return 'No valid data found in hyperspectral .mat file', 400

        data_var_name = var_names[0]
        data_store.hyperspectral_data = mat_contents[data_var_name]

        if len(data_store.hyperspectral_data.shape) != 3:
            return 'Invalid hyperspectral data format. Expected 3D array.', 400
    except Exception as e:
        return f'Error loading hyperspectral data: {str(e)}', 400

    # Save and load label data
    try:
        label_file.save(label_path)
        label_contents = loadmat(label_path)
        label_var_names = [name for name in label_contents.keys() if not name.startswith('__')]
        if len(label_var_names) == 0:
            return 'No valid data found in label .mat file', 400

        label_var_name = label_var_names[0]
        data_store.label_data = label_contents[label_var_name]
    except Exception as e:
        return f'Error loading label file: {str(e)}', 400

    return redirect(url_for('preprocess'))


@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
    if request.method == 'GET':
        if data_store.hyperspectral_data is None:
            return redirect(url_for('index'))
        data_shape = data_store.hyperspectral_data.shape
        return render_template('preprocess.html', data_shape=data_shape, num_bands=data_shape[2])

    try:
        snr_threshold = float(request.form.get('snr_threshold', 0.1))
        n_components = int(request.form.get('n_components', 30))

        data = data_store.hyperspectral_data
        height, width, bands = data.shape

        # 计算 SNR 并剔除低质量波段
        band_means = np.mean(data, axis=(0, 1))
        band_stds = np.std(data, axis=(0, 1)) + 1e-10
        snr = band_means / band_stds
        good_bands = snr > snr_threshold
        filtered_data = data[:, :, good_bands]

        # PCA 降维
        reshaped = filtered_data.reshape(-1, filtered_data.shape[-1])
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        scaler = StandardScaler()
        reshaped = scaler.fit_transform(reshaped)
        pca = PCA(n_components=min(n_components, reshaped.shape[1]))
        reduced = pca.fit_transform(reshaped)

        # Z-score标准化
        mean = np.mean(reduced, axis=0)
        std = np.std(reduced, axis=0) + 1e-8
        normalized = (reduced - mean) / std

        # 恢复为图像格式
        reduced_3d = normalized.reshape(height, width, -1)
        data_store.processed_data = reduced_3d

        # 保存处理结果
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_data.mat')
        savemat(save_path, {
            'processed_data': reduced_3d,
            'pca_components': pca.components_,
            'explained_variance': pca.explained_variance_ratio_
        })

        return redirect(url_for('train'))
    except Exception as e:
        return f'预处理出错: {str(e)}', 400

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'GET':
        if data_store.processed_data is None:
            return redirect(url_for('index'))
        return render_template('train.html', pretrained=True)

    try:
        # 准备数据
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 设置预训练模型路径
        upload_folder = os.path.abspath(app.config['UPLOAD_FOLDER'])
        pretrained_path = os.path.abspath(os.path.join(upload_folder, 'pretrained_model.pt'))

        # 处理输入数据
        X = data_store.processed_data
        height, width, channels = X.shape
        print(f"Data shape: {X.shape}")

        # 改进的数据预处理
        # 1. 标准化每个波段
        X_processed = np.zeros_like(X, dtype=np.float32)
        for i in range(channels):
            band = X[:, :, i]
            mean = np.mean(band)
            std = np.std(band)
            X_processed[:, :, i] = (band - mean) / (std + 1e-8)

        # 2. 全局标准化
        X_processed = (X_processed - np.mean(X_processed)) / (np.std(X_processed) + 1e-8)

        # 打印数据统计信息
        print(f"Processed data stats - Mean: {np.mean(X_processed):.4f}, Std: {np.std(X_processed):.4f}")
        print(f"Data range - Min: {np.min(X_processed):.4f}, Max: {np.max(X_processed):.4f}")

        # 初始化模型（调整参数）
        model = SC_SS_MTr(
            num_bands=channels,
            num_classes=len(np.unique(data_store.label_data)),
            patch_size=1,
            depth=6,  # 减小深度
            drop_rate=0.1  # 添加dropout
        ).to(device)

        # 加载预训练模型
        checkpoint = torch.load(pretrained_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        else:
            pretrained_dict = checkpoint

        # 只加载匹配的权重
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

        model.eval()

        # 生成预测结果
        predictions = np.zeros((height, width), dtype=np.int64)

        # 使用滑动窗口进行预测
        window_size = 16  # 设置滑动窗口大小
        stride = 8  # 设置步长

        with torch.no_grad():
            for i in range(0, height - window_size + 1, stride):
                for j in range(0, width - window_size + 1, stride):
                    window = X_processed[i:i + window_size, j:j + window_size, :]
                    window_tensor = torch.FloatTensor(window).permute(2, 0, 1).unsqueeze(0).to(device)

                    outputs = model(window_tensor)
                    predicted_class = outputs.argmax(dim=1).item()

                    center_i = i + window_size // 2
                    center_j = j + window_size // 2
                    if 0 <= center_i < height and 0 <= center_j < width:
                        predictions[center_i, center_j] = predicted_class

                # 打印进度
                print(f"Processed {i}/{height} rows")

        # 处理边缘区域
        # 填充未预测的边缘像素
        for i in range(height):
            for j in range(width):
                if predictions[i, j] == 0:
                    # 使用最近邻填充
                    valid_neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < height and 0 <= nj < width and predictions[ni, nj] != 0:
                                valid_neighbors.append(predictions[ni, nj])
                    if valid_neighbors:
                        predictions[i, j] = max(set(valid_neighbors), key=valid_neighbors.count)

        # 计算评估指标
        metrics = calculate_metrics(data_store.label_data.flatten(), predictions.flatten())

        # 打印详细的评估指标
        print("\nDetailed Metrics:")
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"Average Accuracy: {metrics['average_accuracy']:.4f}")
        print(f"Kappa Coefficient: {metrics['kappa']:.4f}")
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])

        # 保存结果
        try:
            # 保存分类结果图（使用更好的可视化）
            plt.figure(figsize=(12, 12))
            plt.imshow(predictions, cmap='tab20')  # 使用更好的颜色映射
            plt.colorbar(label='Class')
            plt.title('Classification Result')
            plt.axis('off')
            plt.savefig(os.path.join(upload_folder, 'classification_map.png'),
                        bbox_inches='tight', dpi=300)
            plt.close()

            # 生成更好的混淆矩阵可视化
            plt.figure(figsize=(12, 10))
            sns.heatmap(metrics['confusion_matrix'],
                        annot=True,
                        fmt='d',
                        cmap='YlOrRd',
                        xticklabels=range(len(np.unique(data_store.label_data))),
                        yticklabels=range(len(np.unique(data_store.label_data))))
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Class')
            plt.ylabel('True Class')
            plt.savefig(os.path.join(upload_folder, 'confusion_matrix.png'),
                        bbox_inches='tight', dpi=300)
            plt.close()

            # 保存详细结果
            savemat(os.path.join(upload_folder, 'classification_results.mat'),
                    {'classification_map': predictions,
                     'confusion_matrix': metrics['confusion_matrix'],
                     'overall_accuracy': metrics['overall_accuracy'],
                     'average_accuracy': metrics['average_accuracy'],
                     'kappa': metrics['kappa'],
                     'per_class_accuracy': metrics.get('per_class_accuracy', None)})

        except Exception as e:
            print(f"Error saving results: {str(e)}")
            return f'Error saving results: {str(e)}', 400

        return render_template('results.html',
                               metrics=metrics,
                               pretrained=True)

    except Exception as e:
        print(f"General error: {str(e)}")
        traceback.print_exc()
        return f'Error during processing: {str(e)}', 400


@app.route('/get_classification_map')
def get_classification_map():
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], 'classification_map.png'),
                     mimetype='image/png')


@app.route('/get_confusion_matrix')
def get_confusion_matrix():
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], 'confusion_matrix.png'),
                     mimetype='image/png')


@app.route('/download_results')
def download_results():
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], 'classification_results.mat'),
                     mimetype='application/octet-stream',
                     as_attachment=True,
                     download_name='classification_results.mat')


if __name__ == '__main__':
    app.run(debug=True)