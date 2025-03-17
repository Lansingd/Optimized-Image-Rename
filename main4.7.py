from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QGroupBox, QPushButton, QLabel, QTextEdit, QProgressBar,
                            QRadioButton, QCheckBox, QFileDialog, QMessageBox)
from PySide6.QtCore import Qt, QThread, Signal
import traceback
import sys
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
import torch

class EnhancedOCR:
    def __init__(self, use_easyocr=True, use_paddleocr=True):
        self.use_easyocr = use_easyocr
        self.use_paddleocr = use_paddleocr
        self.gpu_available = self.check_gpu()

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
            return torch.cuda.is_available()
        except:
            return False

    def recognize_text(self, img):
        """返回包含置信度的识别结果"""
        results = {}
        if self.use_easyocr and self.easy_reader:
            try:
                # 批量大小控制，防止显存溢出
                easy_full = self.easy_reader.readtext(img, batch_size=1)
                easy_text = ' '.join([res[1] for res in easy_full])
                easy_conf = sum(res[2] for res in easy_full)/len(easy_full) if easy_full else 0
                results['easyocr'] = (easy_text, easy_conf)
                # 清理显存
                torch.cuda.empty_cache() if self.gpu_available else None
            except Exception as e:
                print(f"EasyOCR 识别异常: {str(e)}")
                results['easyocr'] = ("", 0)

        if self.use_paddleocr and self.paddle_ocr:
            try:
                paddle_result = self.paddle_ocr.ocr(img, cls=True)
                paddle_text = []
                paddle_confs = []
                
                if paddle_result is not None:
                    for line in paddle_result:
                        if line:
                            for word_info in line:
                                if word_info and len(word_info) >= 2:
                                    text = str(word_info[1][0])
                                    conf = float(word_info[1][1])
                                    paddle_text.append(text)
                                    paddle_confs.append(conf)
                
                paddle_text_str = ' '.join(paddle_text) if paddle_text else ""
                paddle_conf_avg = sum(paddle_confs)/len(paddle_confs) if paddle_confs else 0.0
                results['paddleocr'] = (paddle_text_str, paddle_conf_avg)
            except Exception as e:
                print(f"PaddleOCR 识别异常: {str(e)}")
                results['paddleocr'] = ("", 0)
        
        return results

class Worker(QThread):
    update_log = Signal(str)
    update_progress = Signal(int)
    finished = Signal()

    def __init__(self, main_app):
        super().__init__()
        self.main_app = main_app
        self.stop_flag = False

    def run(self):
        ocr_engine = None
        try:
            use_easyocr = self.main_app.radio_easy.isChecked() or self.main_app.radio_both.isChecked()
            use_paddleocr = self.main_app.radio_paddle.isChecked() or self.main_app.radio_both.isChecked()
            ocr_engine = EnhancedOCR(use_easyocr, use_paddleocr)

            name_data = self.main_app.load_name_library(self.main_app.name_file)
            if not name_data[0]:
                return

            name_lib, full_names = name_data
            image_files = [f for f in os.listdir(self.main_app.image_folder)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            total = len(image_files)
            self.update_progress.emit(0)
            self.main_app.progress.setMaximum(total)

            for idx, filename in enumerate(image_files):
                if self.stop_flag:
                    self.update_log.emit("任务已中止")
                    break
                file_path = os.path.join(self.main_app.image_folder, filename)
                self.update_log.emit(f"正在处理: {filename} ({idx + 1}/{total})")

                try:
                    image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError("图像解码失败")

                    process_area = image if self.main_app.check_full.isChecked() else \
                        image[int(image.shape[0] * 0.8):, int(image.shape[1] * 0.6):]
                    
                    if self.main_app.check_preprocess.isChecked():
                        process_area = self.main_app.preprocess_image(process_area)

                    ocr_results = ocr_engine.recognize_text(process_area)
                    
                    # 根据引擎选择显示结果
                    if use_easyocr and 'easyocr' in ocr_results:
                        text, conf = ocr_results['easyocr']
                        self.update_log.emit(f"EasyOCR结果: {text} (置信度: {conf:.2f})")
                    if use_paddleocr and 'paddleocr' in ocr_results:
                        text, conf = ocr_results['paddleocr']
                        self.update_log.emit(f"PaddleOCR结果: {text} (置信度: {conf:.2f})")

                    best_match = None
                    if use_easyocr and 'easyocr' in ocr_results:
                        best_match = self.main_app.find_best_match(ocr_results['easyocr'][0], full_names)
                    
                    if not best_match and use_paddleocr and 'paddleocr' in ocr_results:
                        best_match = self.main_app.find_best_match(ocr_results['paddleocr'][0], full_names)
                    
                    if not best_match and self.main_app.check_single.isChecked():
                        combined_text = ' '.join([v[0] for v in ocr_results.values()])
                        best_match = self.main_app.find_best_match(combined_text, full_names)

                    if best_match:
                        safe_name = re.sub(r'[\\/*?:"<>|]', '#', best_match)[:220]
                        ext = os.path.splitext(filename)[1]
                        new_name = f"{safe_name}{ext}"
                        counter = 1
                        while os.path.exists(os.path.join(self.main_app.image_folder, new_name)):
                            new_name = f"{safe_name}_{counter}{ext}"
                            counter += 1

                        shutil.move(file_path, os.path.join(self.main_app.image_folder, new_name))
                        self.update_log.emit(f"重命名成功: {filename} → {new_name}")

                    self.update_progress.emit(idx + 1)

                except Exception as e:
                    self.update_log.emit(f"处理失败 [{filename}]: {str(e)}")

            self.update_log.emit("处理完成！" if not self.stop_flag else "处理已中止")

        except Exception as e:
            error_msg = f"严重错误: {str(e)}\n{traceback.format_exc()}"
            self.update_log.emit(error_msg)
        finally:
            if ocr_engine:
                del ocr_engine
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.finished.emit()

class ImageRenamerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("一画室智能图片命名 v4.7")
        self.setup_ui()
        self.setup_vars()
        self.setup_connections()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

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
        
        ocr_group = QGroupBox("识别设置")
        ocr_layout = QVBoxLayout(ocr_group)
        
        self.radio_easy = QRadioButton("EasyOCR")
        self.radio_paddle = QRadioButton("PaddleOCR")
        self.radio_both = QRadioButton("双重识别")
        self.radio_both.setChecked(True)
        
        engine_layout = QHBoxLayout()
        engine_layout.addWidget(QLabel("OCR引擎:"))
        engine_layout.addWidget(self.radio_easy)
        engine_layout.addWidget(self.radio_paddle)
        engine_layout.addWidget(self.radio_both)
        
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
        
        log_group = QGroupBox("处理日志")
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout = QVBoxLayout(log_group)
        log_layout.addWidget(self.log_text)
        
        self.progress = QProgressBar()
        self.progress.setAlignment(Qt.AlignCenter)
        
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始处理")
        self.stop_btn = QPushButton("停止")
        self.exit_btn = QPushButton("退出")
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.exit_btn)
        
        main_layout.addWidget(file_group)
        main_layout.addWidget(ocr_group)
        main_layout.addWidget(log_group)
        main_layout.addWidget(self.progress)
        main_layout.addLayout(btn_layout)

    def setup_vars(self):
        self.image_folder = ""
        self.name_file = ""
        self.similar_map = self.load_similar_map()

    def load_similar_map(self):
        similar_map = {}
        try:
            with open("similar_map.txt", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and ":" in line:
                        key, value = line.split(":", 1)
                        similar_map[key.strip()] = value.strip()
            self.log(f"已加载 {len(similar_map)} 条形近字映射")
        except Exception:
            self.log("未找到形近字映射文件，使用默认匹配")
        return similar_map

    def setup_connections(self):
        self.image_btn.clicked.connect(self.select_image_folder)
        self.name_btn.clicked.connect(self.select_name_file)
        self.start_btn.clicked.connect(self.start_processing)
        self.stop_btn.clicked.connect(self.stop_processing)
        self.exit_btn.clicked.connect(self.close)

    def log(self, message):
        self.log_text.append(f"{time.strftime('%H:%M:%S')} [OCR] - {message}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def select_image_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder:
            self.image_folder = folder
            self.image_label.setText(folder)
            self.log(f"已选择图片文件夹: {folder}")

    def select_name_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "选择姓名库文件", "", "文本文件 (*.txt)")
        if filepath:
            self.name_file = filepath
            self.name_label.setText(filepath)
            self.log(f"已选择姓名库文件: {filepath}")

    def start_processing(self):
        # 检查必要的文件路径是否已选择
        if not self.image_folder or not self.name_file:
            QMessageBox.critical(self, "错误", "请先选择图片文件夹和姓名库文件")
            return
        # 添加日志信息，提示用户引擎正在启动
        self.log("正在启动所选引擎")
        # 创建并启动工作线程
        self.worker = Worker(self)
        self.worker.update_log.connect(self.log)
        self.worker.update_progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def stop_processing(self):
        if hasattr(self, 'worker'):
            self.worker.stop_flag = True
            self.log("正在停止当前任务...")
            self.worker.wait()
            self.log("任务已停止")

    def update_progress(self, value):
        self.progress.setValue(value)

    def on_finished(self):
        self.log("任务完成！")

    def load_name_library(self, path):
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
                                name_dict[name[i:j]].append(name)
            self.log(f"成功加载 {len(full_names)} 个姓名")
            return name_dict, full_names
        except Exception as e:
            self.log(f"加载姓名库失败: {str(e)}")
            return None, None

    def find_best_match(self, text, full_names):
        try:
            clean_text = ''.join([c for c in str(text) if '\u4e00' <= c <= '\u9fff'])
            modified_text = self.apply_similar_replace(clean_text)
            return self._do_match(modified_text, full_names) or self._do_match(clean_text, full_names)
        except Exception as e:
            self.log(f"匹配异常: {str(e)}")
            return None

    def apply_similar_replace(self, text):
        clean_text = ''.join([c for c in str(text) if '\u4e00' <= c <= '\u9fff'])
        replaced = []
        i = 0
        while i < len(clean_text):
            matched = False
            if i + 1 < len(clean_text):
                pair = clean_text[i:i+2]
                if pair in self.similar_map:
                    replaced.append(self.similar_map[pair])
                    i += 2
                    matched = True
            if not matched and i < len(clean_text):
                single = clean_text[i]
                replaced.append(self.similar_map.get(single, single))
                i += 1
        return ''.join(replaced)

    def _do_match(self, clean_text, full_names):
        for name in full_names:
            if name in clean_text:
                return name

        best_match = None
        max_length = 0
        for name in full_names:
            match = SequenceMatcher(None, clean_text, name).find_longest_match()
            if match.size > max_length:
                max_length = match.size
                best_match = name
        if max_length >= 2:
            return best_match

        if self.check_single.isChecked():
            char_counts = defaultdict(int)
            for c in clean_text:
                for name in full_names:
                    if c in name:
                        char_counts[name] += 1
            if char_counts:
                return max(char_counts.items(), key=lambda x: (x[1], full_names.index(x[0])))[0]
        return None

    def preprocess_image(self, img):
        try:
            if not isinstance(img, np.ndarray):
                img = np.array(img, dtype=np.uint8)
            elif img.dtype != np.uint8:
                img = img.astype(np.uint8)
                
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            processed = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            return cv2.GaussianBlur(processed, (3, 3), 0)
        except Exception as e:
            self.log(f"图像预处理异常: {str(e)}")
            return img

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageRenamerApp()
    window.show()
    sys.exit(app.exec())