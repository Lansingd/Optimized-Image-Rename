from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QGroupBox, QPushButton, QLabel, QTextEdit, QProgressBar,
                            QRadioButton, QCheckBox, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import os
import cv2
import easyocr
import numpy as np
import re
import shutil
import time
from collections import defaultdict
from difflib import SequenceMatcher
from paddleocr import PaddleOCR

class EnhancedOCR:
    def __init__(self, use_easyocr=True, use_paddleocr=True):
        self.use_easyocr = use_easyocr
        self.use_paddleocr = use_paddleocr
        self.gpu_available = self.check_gpu()

        # 初始化OCR引擎
        self.easy_reader = None
        self.paddle_ocr = None

        if self.use_easyocr:
            self.easy_reader = easyocr.Reader(
                ['ch_sim'],
                gpu=self.gpu_available,
                model_storage_directory='./models',
                download_enabled=True
            )

        if self.use_paddleocr:
            self.paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang="ch",
                use_gpu=self.gpu_available,
                show_log=False
            )

    def check_gpu(self):
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def recognize_text(self, img):
        texts = []
        if self.use_easyocr and self.easy_reader:
            try:
                easy_result = self.easy_reader.readtext(img, detail=0)
                texts.append(' '.join(easy_result) if easy_result else "")
            except Exception as e:
                print(f"EasyOCR 识别异常: {str(e)}")

        if self.use_paddleocr and self.paddle_ocr:
            try:
                paddle_result = self.paddle_ocr.ocr(img, cls=True) or [[]]
                texts.append(' '.join([line[1][0] for line in paddle_result[0]]) if paddle_result[0] else "")
            except Exception as e:
                print(f"PaddleOCR 识别异常: {str(e)}")

        return self._select_best_result(texts)

    def _select_best_result(self, texts):
        """改进的结果选择策略"""
        def chinese_count(text):
            return len([c for c in text if '\u4e00' <= c <= '\u9fff'])

        valid_texts = [t for t in texts if 2 <= chinese_count(t) <= 4]
        if valid_texts:
            return max(valid_texts, key=len)
        return max(texts, key=len, default="")

class Worker(QThread):
    update_log = pyqtSignal(str)
    update_progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, main_app):
        super().__init__()
        self.main_app = main_app
        self.stop_flag = False

    def run(self):
        try:
            # 初始化OCR引擎
            use_easyocr = self.main_app.radio_easy.isChecked() or self.main_app.radio_both.isChecked()
            use_paddleocr = self.main_app.radio_paddle.isChecked() or self.main_app.radio_both.isChecked()
            ocr_engine = EnhancedOCR(use_easyocr, use_paddleocr)

            # 加载姓名库
            name_data = self.main_app.load_name_library(self.main_app.name_file)
            if not name_data[0]:
                return

            name_lib, full_names = name_data
            image_files = [f for f in os.listdir(self.main_app.image_folder)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            total = len(image_files)
            self.update_progress.emit(0)  # 初始化进度条
            self.main_app.progress.setMaximum(total)  # 设置进度条最大值

            for idx, filename in enumerate(image_files):
                if self.stop_flag:
                    break

                file_path = os.path.join(self.main_app.image_folder, filename)
                self.update_log.emit(f"正在处理: {filename} ({idx + 1}/{total})")

                try:
                    # 读取图像
                    image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError("图像解码失败")

                    # 选择识别区域
                    if self.main_app.check_full.isChecked():
                        process_area = image
                    else:
                        h, w = image.shape[:2]
                        process_area = image[int(h * 0.8):h, int(w * 0.6):w]  # 右下角区域

                    # 图像预处理
                    if self.main_app.check_preprocess.isChecked():
                        process_area = self.main_app.preprocess_image(process_area)

                    # OCR识别
                    ocr_text = ocr_engine.recognize_text(process_area)
                    best_match = self.main_app.find_best_match(ocr_text, full_names)

                    if best_match:
                        # 生成安全文件名
                        safe_name = re.sub(r'[\\/*?:"<>|]', '#', best_match)[:220]
                        ext = os.path.splitext(filename)[1]
                        new_name = f"{safe_name}{ext}"

                        # 处理重名
                        counter = 1
                        while os.path.exists(os.path.join(self.main_app.image_folder, new_name)):
                            new_name = f"{safe_name}_{counter}{ext}"
                            counter += 1

                        shutil.move(file_path, os.path.join(self.main_app.image_folder, new_name))
                        self.update_log.emit(f"重命名成功: {filename} → {new_name}")

                    # 更新进度
                    self.update_progress.emit(idx + 1)

                except Exception as e:
                    self.update_log.emit(f"处理失败 [{filename}]: {str(e)}")

            self.update_log.emit("处理完成！" if not self.stop_flag else "处理已中止")

        except Exception as e:
            self.update_log.emit(f"发生严重错误: {str(e)}")
        finally:
            self.finished.emit()

class ImageRenamerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("一画室智能图片命名 v4.5")
        self.setup_ui()
        self.setup_vars()
        self.setup_connections()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # 文件设置区域
        file_group = QGroupBox("文件设置")
        file_layout = QVBoxLayout(file_group)
        
        self.image_btn = QPushButton("选择图片文件夹")
        self.image_label = QLabel("未选择")
        self.name_btn = QPushButton("选择姓名库文件")
        self.name_label = QLabel("未选择")
        
        file_layout.addWidget(self.image_btn)
        file_layout.addWidget(self.image_label)
        file_layout.addWidget(self.name_btn)
        file_layout.addWidget(self.name_label)
        
        # OCR设置区域
        ocr_group = QGroupBox("识别设置")
        ocr_layout = QVBoxLayout(ocr_group)
        
        # 引擎选择
        self.radio_easy = QRadioButton("EasyOCR")
        self.radio_paddle = QRadioButton("PaddleOCR")
        self.radio_both = QRadioButton("双重识别")
        self.radio_both.setChecked(True)
        
        engine_layout = QHBoxLayout()
        engine_layout.addWidget(QLabel("OCR引擎:"))
        engine_layout.addWidget(self.radio_easy)
        engine_layout.addWidget(self.radio_paddle)
        engine_layout.addWidget(self.radio_both)
        
        # 高级选项
        self.check_single = QCheckBox("单字兜底匹配")
        self.check_preprocess = QCheckBox("启用图像预处理")
        self.check_full = QCheckBox("全图识别模式")
        self.check_preprocess.setChecked(True)
        
        option_layout = QHBoxLayout()
        option_layout.addWidget(QLabel("高级设置:"))
        option_layout.addWidget(self.check_single)
        option_layout.addWidget(self.check_preprocess)
        option_layout.addWidget(self.check_full)
        
        ocr_layout.addLayout(engine_layout)
        ocr_layout.addLayout(option_layout)
        
        # 日志区域
        log_group = QGroupBox("处理日志")
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout = QVBoxLayout(log_group)
        log_layout.addWidget(self.log_text)
        
        # 进度条
        self.progress = QProgressBar()
        self.progress.setAlignment(Qt.AlignCenter)
        
        # 操作按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始处理")
        self.stop_btn = QPushButton("停止")
        self.exit_btn = QPushButton("退出")
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.exit_btn)
        
        # 组合所有组件
        main_layout.addWidget(file_group)
        main_layout.addWidget(ocr_group)
        main_layout.addWidget(log_group)
        main_layout.addWidget(self.progress)
        main_layout.addLayout(btn_layout)


    def setup_vars(self):
        """初始化变量"""
        self.image_folder = ""
        self.name_file = ""
        self.similar_map = self.load_similar_map()
        
    def load_similar_map(self):
        """加载形近字映射表"""
        similar_map = {}
        try:
            with open("similar_map.txt", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and ":" in line:
                        key, value = line.split(":", 1)
                        similar_map[key.strip().strip('"')] = value.strip().strip('",')
            self.log(f"已加载 {len(similar_map)} 条形近字映射")
        except Exception as e:
            self.log(f"加载形近字映射失败: {str(e)}")
        return similar_map

    def setup_connections(self):
        self.image_btn.clicked.connect(self.select_image_folder)
        self.name_btn.clicked.connect(self.select_name_file)
        self.start_btn.clicked.connect(self.start_processing)
        self.stop_btn.clicked.connect(self.stop_processing)
        self.exit_btn.clicked.connect(self.close)

    def log(self, message):
        """线程安全的日志记录"""
        self.log_text.append(f"{time.strftime('%H:%M:%S')} - {message}")

    def update_progress(self, value):
        self.progress.setValue(value)

    def on_finished(self):
        self.log("任务已完成或中止")

    def select_image_folder(self):
        """选择图片文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder:
            self.image_folder = folder
            self.image_label.setText(folder)
            self.log(f"已选择图片文件夹: {folder}")

    def select_name_file(self):
        """选择姓名库文件"""
        filepath, _ = QFileDialog.getOpenFileName(self, "选择姓名库文件", "", "文本文件 (*.txt)")
        if filepath:
            self.name_file = filepath
            self.name_label.setText(filepath)
            self.log(f"已选择姓名库文件: {filepath}")

    def start_processing(self):
        """启动处理线程"""
        if not self.image_folder or not self.name_file:
            QMessageBox.critical(self, "错误", "请先选择图片文件夹和姓名库文件")
            return
        self.worker = Worker(self)
        self.worker.update_log.connect(self.log)
        self.worker.update_progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def stop_processing(self):
        if hasattr(self, 'worker'):
            self.worker.stop_flag = True
            self.log("正在停止当前任务...")
            
    def update_progress(self, value):
        """更新进度条"""
        self.progress.setValue(value)

    def on_finished(self):
        """任务完成后的处理"""
        self.log("任务完成！")

    def load_name_library(self, path):
        """加载姓名库"""
        name_dict = defaultdict(list)
        full_names = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    name = line.strip()
                    if 2 <= len(name) <= 4:
                        full_names.append(name)
                        for i in range(len(name)):
                            for j in range(i + 1, len(name) + 1):
                                key = name[i:j]
                                name_dict[key].append(name)
            self.log(f"成功加载 {len(full_names)} 个姓名")
            return name_dict, full_names
        except Exception as e:
            self.log(f"加载姓名库失败: {str(e)}")
            return None, None

    def find_best_match(self, text, full_names):
        """增强型匹配逻辑"""
        clean_text = ''.join([c for c in text if '\u4e00' <= c <= '\u9fff'])

        # 阶段1：形近字匹配
        for pattern, name in self.similar_map.items():
            if pattern in clean_text and name in full_names:
                return name

        # 阶段2：全名匹配
        for name in full_names:
            if name in clean_text:
                return name

        # 阶段3：子串匹配
        best_match = None
        max_length = 0
        for name in full_names:
            match = SequenceMatcher(None, clean_text, name).find_longest_match()
            if match.size > max_length:
                max_length = match.size
                best_match = name
        if max_length >= 2:
            return best_match

        # 阶段4：单字兜底（可选）
        if self.check_single.isChecked():
            char_counts = defaultdict(int)
            for c in clean_text:
                for name in full_names:
                    if c in name:
                        char_counts[name] += 1
            if char_counts:
                max_count = max(char_counts.values())
                candidates = [k for k, v in char_counts.items() if v == max_count]
                return sorted(candidates, key=lambda x: full_names.index(x))[0]

        return None

    def preprocess_image(self, img):
        """可配置的图像预处理"""
        # 灰底白字专用预处理
        inverted = cv2.bitwise_not(img)
        alpha = 1.8
        beta = -60
        adjusted = cv2.convertScaleAbs(inverted, alpha=alpha, beta=beta)
        gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (3, 3), 0)

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    window = ImageRenamerApp()
    window.show()
    sys.exit(app.exec_())