from flask import Flask, render_template, request, redirect, url_for, send_file, session, flash
import numpy as np
from scipy.io import loadmat, savemat
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import matplotlib
import traceback

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from utils.metrics import calculate_metrics
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from datetime import datetime
import shutil

app = Flask(__name__, static_folder='static')
app.secret_key = 'getain_hyperspectral_processing_2025'  # 使用安全的密钥

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB


# 全局数据存储类
class DataStore:
    def __init__(self):
        self.reset()

    def reset(self):
        self.hyperspectral_data = None
        self.label_data = None
        self.processed_data = None
        self.step_completed = {
            'upload': False,
            'preprocess': False,
            'train': False
        }
        # 清理上传文件夹
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f'Error deleting {file_path}: {e}')


data_store = DataStore()


@app.route('/')
def index():
    # 重置数据存储
    data_store.reset()
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'hyperspectral' not in request.files or 'label' not in request.files:
        flash('请上传高光谱数据文件和标签文件')
        return redirect(url_for('index'))

    hyperspectral_file = request.files['hyperspectral']
    label_file = request.files['label']

    if hyperspectral_file.filename == '' or label_file.filename == '':
        flash('请选择文件')
        return redirect(url_for('index'))

    if not (hyperspectral_file.filename.endswith('.mat') and label_file.filename.endswith('.mat')):
        flash('请上传.mat格式的文件')
        return redirect(url_for('index'))

    try:
        # 保存文件
        hyperspectral_path = os.path.join(app.config['UPLOAD_FOLDER'], 'hyperspectral.mat')
        label_path = os.path.join(app.config['UPLOAD_FOLDER'], 'label.mat')

        # 加载并验证高光谱数据
        hyperspectral_file.save(hyperspectral_path)
        mat_contents = loadmat(hyperspectral_path)
        var_names = [name for name in mat_contents.keys() if not name.startswith('__')]
        if len(var_names) == 0:
            raise ValueError('高光谱数据文件中未找到有效数据')

        data_var_name = var_names[0]
        data_store.hyperspectral_data = mat_contents[data_var_name]

        if len(data_store.hyperspectral_data.shape) != 3:
            raise ValueError('高光谱数据格式错误，需要三维数组（高度×宽度×波段数）')

        # 加载并验证标签数据
        label_file.save(label_path)
        label_contents = loadmat(label_path)
        label_var_names = [name for name in label_contents.keys() if not name.startswith('__')]
        if len(label_var_names) == 0:
            raise ValueError('标签文件中未找到有效数据')

        label_var_name = label_var_names[0]
        data_store.label_data = label_contents[label_var_name]

        # 验证数据尺寸匹配
        if data_store.label_data.shape[:2] != data_store.hyperspectral_data.shape[:2]:
            raise ValueError('标签数据尺寸与高光谱图像尺寸不匹配')

        # 设置上传步骤完成标志
        data_store.step_completed['upload'] = True
        flash('文件上传成功！')
        return redirect(url_for('preprocess'))

    except Exception as e:
        flash(f'上传文件时出错: {str(e)}')
        return redirect(url_for('index'))


@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
    if not data_store.step_completed['upload']:
        flash('请先上传数据文件')
        return redirect(url_for('index'))

    if request.method == 'GET':
        data_shape = data_store.hyperspectral_data.shape
        return render_template('preprocess.html',
                               data_shape=data_shape,
                               num_bands=data_shape[2])

    try:
        snr_threshold = float(request.form.get('snr_threshold', 0.1))
        n_components = int(request.form.get('n_components', 30))

        if not (0 < snr_threshold <= 1):
            raise ValueError('信噪比阈值必须在0到1之间')

        data = data_store.hyperspectral_data
        height, width, bands = data.shape

        if not (1 <= n_components <= bands):
            raise ValueError(f'PCA组件数必须在1到{bands}之间')

        # 计算SNR并剔除低质量波段
        band_means = np.mean(data, axis=(0, 1))
        band_stds = np.std(data, axis=(0, 1)) + 1e-10
        snr = band_means / band_stds
        good_bands = snr > snr_threshold
        filtered_data = data[:, :, good_bands]

        if filtered_data.shape[2] == 0:
            raise ValueError('SNR阈值过高，所有波段都被剔除')

        # PCA降维
        reshaped = filtered_data.reshape(-1, filtered_data.shape[-1])
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

        # 设置预处理步骤完成标志
        data_store.step_completed['preprocess'] = True
        flash('数据预处理完成！')
        return redirect(url_for('train'))

    except Exception as e:
        flash(f'预处理过程中出错: {str(e)}')
        return redirect(url_for('preprocess'))


@app.route('/train', methods=['GET', 'POST'])
def train():
    # 检查前置步骤
    if not data_store.step_completed['upload']:
        flash('请先上传数据文件')
        return redirect(url_for('index'))

    if not data_store.step_completed['preprocess']:
        flash('请先完成数据预处理步骤')
        return redirect(url_for('preprocess'))

    if request.method == 'GET':
        return render_template('train.html')

    try:
        # 验证数据完整性
        if data_store.processed_data is None:
            raise ValueError("请先完成数据预处理步骤")
        if data_store.label_data is None:
            raise ValueError("请先上传标签文件")

        # 准备数据
        X = data_store.processed_data
        y = data_store.label_data
        height, width, n_features = X.shape

        # 确保标签数据与图像尺寸匹配
        if y.shape[:2] != (height, width):
            raise ValueError(f"标签数据维度 ({y.shape}) 与图像维度 ({height}, {width}) 不匹配")

        # 重塑数据为2D形式
        X_2d = X.reshape(-1, n_features)
        y_2d = y.reshape(-1)

        # 获取有效像素（包括背景）
        valid_indices = np.arange(len(y_2d))
        background_mask = y_2d == 0

        # 分离背景和非背景像素
        X_nonbg = X_2d[~background_mask]
        y_nonbg = y_2d[~background_mask]

        # 检查是否有足够的非背景像素
        if len(X_nonbg) == 0:
            raise ValueError("数据中没有找到非背景像素")

        # 数据集分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_nonbg, y_nonbg,
            test_size=0.3,
            random_state=42,
            stratify=y_nonbg
        )

        # 训练SVM模型
        svm = SVC(
            kernel='rbf',
            C=10.0,
            gamma='auto',
            random_state=42,
            probability=True
        )
        svm.fit(X_train, y_train)

        # 评估测试集
        test_predictions = svm.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)

        # 预测整个图像
        predictions_2d = np.zeros_like(y_2d)
        batch_size = 10000

        # 对非背景像素进行预测
        non_bg_indices = np.where(~background_mask)[0]
        for i in range(0, len(non_bg_indices), batch_size):
            batch_indices = non_bg_indices[i:i + batch_size]
            batch_predictions = svm.predict(X_2d[batch_indices])
            predictions_2d[batch_indices] = batch_predictions

        # 重塑回原始图像形状
        predictions = predictions_2d.reshape(height, width)

        # 计算评估指标
        metrics = calculate_metrics(y_2d[~background_mask], predictions_2d[~background_mask])

        # 保存结果
        try:
            # 保存分类结果图
            plt.figure(figsize=(12, 12))
            unique_classes = len(np.unique(predictions))
            cmap = plt.cm.get_cmap('tab20', unique_classes)
            im = plt.imshow(predictions, cmap=cmap)
            plt.colorbar(im, label='类别')
            plt.title('分类结果图')
            plt.axis('off')
            plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'classification_map.png'),
                        bbox_inches='tight', dpi=300)
            plt.close()

            # 生成混淆矩阵可视化
            plt.figure(figsize=(12, 10))
            cm = metrics['confusion_matrix']
            sns.heatmap(cm,
                        annot=True,
                        fmt='d',
                        cmap='YlOrRd',
                        xticklabels=range(1, len(np.unique(y_nonbg)) + 1),
                        yticklabels=range(1, len(np.unique(y_nonbg)) + 1))
            plt.title('混淆矩阵')
            plt.xlabel('预测类别')
            plt.ylabel('真实类别')
            plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'confusion_matrix.png'),
                        bbox_inches='tight', dpi=300)
            plt.close()

            # 保存详细结果
            savemat(os.path.join(app.config['UPLOAD_FOLDER'], 'classification_results.mat'),
                    {'classification_map': predictions,
                     'confusion_matrix': cm,
                     'overall_accuracy': metrics['overall_accuracy'],
                     'average_accuracy': metrics['average_accuracy'],
                     'kappa': metrics['kappa'],
                     'per_class_accuracy': metrics.get('per_class_accuracy', None),
                     'test_accuracy': test_accuracy})

            # 设置训练步骤完成标志
            data_store.step_completed['train'] = True
            flash('模型训练和预测完成！')
            return render_template('results.html',
                                   metrics=metrics,
                                   test_accuracy=test_accuracy)

        except Exception as e:
            raise ValueError(f'保存结果时出错: {str(e)}')

    except Exception as e:
        error_message = str(e)
        flash(f'处理过程中出错: {error_message}')
        return redirect(url_for('train'))


@app.route('/get_classification_map')
def get_classification_map():
    try:
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], 'classification_map.png'),
                         mimetype='image/png',
                         as_attachment=False)
    except Exception as e:
        flash('获取分类结果图失败')
        return redirect(url_for('train'))


@app.route('/get_confusion_matrix')
def get_confusion_matrix():
    try:
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], 'confusion_matrix.png'),
                         mimetype='image/png',
                         as_attachment=False)
    except Exception as e:
        flash('获取混淆矩阵图失败')
        return redirect(url_for('train'))


@app.route('/download_results')
def download_results():
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], 'classification_results.mat'),
                         mimetype='application/octet-stream',
                         as_attachment=True,
                         download_name=f'classification_results_{timestamp}.mat')
    except Exception as e:
        flash('下载结果文件失败')
        return redirect(url_for('train'))


@app.errorhandler(404)
def page_not_found(e):
    flash('页面未找到')
    return redirect(url_for('index'))


@app.errorhandler(500)
def internal_server_error(e):
    flash('服务器内部错误')
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)