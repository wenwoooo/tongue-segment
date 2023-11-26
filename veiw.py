import os

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QHBoxLayout, QComboBox, QFileDialog
from PyQt5.QtGui import QPixmap
import sys
from PyQt5.QtWidgets import QMessageBox
import subprocess

from analysis import image_analysis


class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 400, 400)  # 设置窗口大小和位置
        self.setWindowTitle('舌体分割分析窗口')  # 设置窗口标题

        layout = QVBoxLayout()  # 垂直布局管理器

        # 创建一个按钮
        button = QPushButton('选择照片')
        button.clicked.connect(self.showImage)  # 连接按钮点击事件到showImage方法
        layout.addWidget(button)  # 将按钮添加到布局中

        # 创建一个标签用来展示图片
        self.label = QLabel(self)
        self.label.setPixmap(QPixmap('image.jpg'))  # 设置标签的背景图片为image.jpg
        layout.addWidget(self.label)  # 将标签添加到布局中

        self.setLayout(layout)  # 设置窗口的主布局
        self.show()  # 显示窗口

    def showImage(self):
        fname, _ = QFileDialog.getOpenFileName(None, 'Open file', '.',
                                               'Image files(*.jpg *.gif *.png)')  # 打开文件选择对话框，选择图片文件
        if fname:  # 如果用户选择了文件
            # 获取文件名（不包括路径和扩展名）
            filename = os.path.basename(fname)
            # 构造标签文件的路径（假设在同一目录下，名字与图片文件相同，只是扩展名为.txt）
            label_fname = f"C:/Users/wen/Desktop/tiaoshibanben/tongue/train_image/{filename}"
            img = QPixmap(label_fname)  # 使用QPixmap加载图片文件并显示在label上。你可能需要根据自己的需求来修改这部分代码。
            self.label.setPixmap(img)
            print("File selected: " + fname)  # 打印选择的文件路径
            print('success load image')  # 你可能需要根据自己需求添加处理错误或未选择的代码。
            load=image_analysis(fname)
            QMessageBox.information(self, "Success", f"Successfully loaded image\n{load}", QMessageBox.Ok)
        else:  # 如果用户没有选择任何文件，那么fname会是None。在这种情况下，我们打印一条消息表示没有加载图片。你可能需要根据自己需求添加处理错误或未选择的代码。
            print('not success load image')  # 你可能需要根据自己需求添加处理错误或未选择的代码




if __name__ == '__main__':
    app = QApplication(sys.argv)  # 创建应用程序实例
    ex = Example()  # 创建窗口实例
    sys.exit(app.exec_())  # 运行应用程序，并监听事件循环
