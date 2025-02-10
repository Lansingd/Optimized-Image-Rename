from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QGroupBox, QPushButton, QLabel, QTextEdit, QProgressBar,
                            QRadioButton, QCheckBox, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
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
        """返回包含置信度的识别结果"""
        results = {}
        if self.use_easyocr and self.easy_reader:
            try:
                easy_full = self.easy_reader.readtext(img)
                easy_text = ' '.join([res[1] for res in easy_full])
                easy_conf = sum(res[2] for res in easy_full)/len(easy_full) if easy_full else 0
                results['easyocr'] = (easy_text, easy_conf)
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
                        if line:  # 过滤空行
                            for word_info in line:
                                if word_info and len(word_info) >= 2:
                                    # 添加类型安全转换
                                    text = str(word_info[1][0])  # 确保文本为字符串
                                    conf = float(word_info[1][1])  # 强制转换为浮点数
                                    paddle_text.append(text)
                                    paddle_confs.append(conf)
                
                paddle_text_str = ' '.join(paddle_text) if paddle_text else ""
                paddle_conf_avg = sum(paddle_confs)/len(paddle_confs) if paddle_confs else 0.0
                results['paddleocr'] = (paddle_text_str, paddle_conf_avg)
            except Exception as e:
                print(f"PaddleOCR 识别异常: {str(e)}")
                results['paddleocr'] = ("", 0)
        
        return results

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

                    # OCR识别（修改部分）
                    # 修改后的匹配逻辑
                    ocr_results = ocr_engine.recognize_text(process_area)
                    easy_text, easy_conf = ocr_results.get('easyocr', ("", 0))
                    paddle_text, paddle_conf = ocr_results.get('paddleocr', ("", 0))
            
                    matches = []
                    if use_easyocr:
                        easy_match = self.main_app.find_best_match(easy_text, full_names)
                        if easy_match: matches.append(('easyocr', easy_match, easy_conf))
                    
                    if use_paddleocr:
                        paddle_match = self.main_app.find_best_match(paddle_text, full_names)
                        if paddle_match: matches.append(('paddleocr', paddle_match, paddle_conf))
            
                    # 结果决策逻辑
                    if len(matches) > 0:
                        # 优先选择置信度高的结果
                        best = max(matches, key=lambda x: x[2])
                        # 当置信度相同时，选择出现更早的结果
                        if best[2] == 0:
                            best = matches[0]
                        best_match = best[1]
                    self.update_log.emit(f"EasyOCR结果: {easy_text} (置信度: {easy_conf:.2f})")
                    self.update_log.emit(f"PaddleOCR结果: {paddle_text} (置信度: {paddle_conf:.2f})")
    
                    best_match = None
                    # 优先检查EasyOCR结果
                    if use_easyocr and ocr_results.get('easyocr'):
                        best_match = self.main_app.find_best_match(ocr_results['easyocr'], full_names)
                    
                    # 如果未匹配成功，检查PaddleOCR结果
                    if not best_match and use_paddleocr and ocr_results.get('paddleocr'):
                        best_match = self.main_app.find_best_match(ocr_results['paddleocr'], full_names)
    
                    # 如果开启单字兜底仍未匹配
                    if not best_match and self.main_app.check_single.isChecked():
                        combined_text = ' '.join([v for v in ocr_results.values()])
                        best_match = self.main_app.find_best_match(combined_text, full_names)
    
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
            error_msg = f"严重错误: {str(e)}\n{traceback.format_exc()}"
            self.update_log.emit(error_msg)
        finally:
            self.finished.emit()

class ImageRenamerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("一画室智能图片命名 v4.6.1")
        self.setup_ui()
        self.setup_vars()
        self.setup_connections()

    # 更新引擎选择判断
    def setup_connections(self):
        # 添加单选按钮互斥
        self.radio_easy.clicked.connect(lambda: self.radio_both.setChecked(False))
        self.radio_paddle.clicked.connect(lambda: self.radio_both.setChecked(False))
        self.radio_both.clicked.connect(lambda: [self.radio_easy.setChecked(False), 
                                                self.radio_paddle.setChecked(False)])

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
        """加载形近字映射表（支持1-2个字符）"""
        similar_map = {}
        try:
            with open("similar_map.txt", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip()
                        value = value.strip()
                        similar_map[key] = value
            self.log(f"已加载 {len(similar_map)} 条形近字映射（支持1-2字符）")
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
        """显示带引擎标识的日志"""
        self.log_text.append(f"{time.strftime('%H:%M:%S')} [OCR] - {message}")

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
        """增强型匹配逻辑（支持形近字修正后重新匹配）"""
        try:
            # 强制转换为字符串并过滤非中文字符
            text_str = str(text)
            clean_text = ''.join([c for c in text_str if '\u4e00' <= c <= '\u9fff'])
            modified_text = self.apply_similar_replace(clean_text)
            
            # 增加调试日志
            self.log(f"原始文本: {text_str} | 清洗后: {clean_text} | 修正后: {modified_text}")
            
            return self._do_match(modified_text, full_names) or self._do_match(clean_text, full_names)
        except Exception as e:
            self.log(f"匹配过程中发生异常: {str(e)}")
            return None
    
    def apply_similar_replace(self, text):
        """应用形近字替换（支持多字符）"""
        try:
            clean_text = ''.join([str(c) for c in text if '\u4e00' <= c <= '\u9fff'])  # 强制转为字符串
            replaced = []
            i = 0
            while i < len(clean_text):
                # 优先匹配最长可能的组合（最多3字符）
                max_len = min(3, len(clean_text)-i)
                found = False
                for l in range(max_len, 0, -1):
                    substr = clean_text[i:i+l]
                    if substr in self.similar_map:
                        replaced.append(str(self.similar_map[substr]))
                        i += l
                        found = True
                        break
                if not found:
                    replaced.append(str(clean_text[i]))
                    i += 1
            return ''.join(replaced)
        except Exception as e:
            self.log(f"形近字替换异常: {str(e)}")
            return str(text)

    def apply_similar_replace(self, text):
        clean_text = ''.join([c for c in str(text) if '\u4e00' <= c <= '\u9fff'])  # 强制转为字符串
        replaced = []
        i = 0
        while i < len(clean_text):
            matched = False
            
            # 添加边界检查
            if i+1 < len(clean_text):
                pair = clean_text[i:i+2]
                if pair in self.similar_map:
                    replaced.append(str(self.similar_map[pair]))  # 确保替换值为字符串
                    i += 2
                    matched = True
            
            if not matched and i < len(clean_text):
                single = str(clean_text[i])  # 强制转为字符串
                replaced.append(str(self.similar_map.get(single, single)))  # 双重字符串转换
                i += 1
        
        return ''.join(replaced)

    def _do_match(self, clean_text, full_names):
        """实际匹配逻辑"""
        # 阶段1：全名匹配
        for name in full_names:
            if name in clean_text:
                return name

        # 阶段2：子串匹配
        best_match = None
        max_length = 0
        for name in full_names:
            match = SequenceMatcher(None, clean_text, name).find_longest_match()
            if match.size > max_length:
                max_length = match.size
                best_match = name
        if max_length >= 2:
            return best_match

        # 阶段3：单字兜底
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
        try:
            # 深度类型校验
            if not isinstance(img, np.ndarray):
                img = np.array(img, dtype=np.uint8)
            elif img.dtype != np.uint8:
                img = img.astype(np.uint8)
                
            # 通道数处理
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:  # 处理RGBA图像
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                
            # 增强对比度
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            processed = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
            return cv2.GaussianBlur(processed, (3, 3), 0)
        except Exception as e:
            self.log(f"图像预处理异常: {str(e)}")
            return img  # 返回原始图像作为兜底

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    window = ImageRenamerApp()
    window.show()
    sys.exit(app.exec_())