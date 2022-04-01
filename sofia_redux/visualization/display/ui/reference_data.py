# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'reference_data.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(644, 234)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.loaded_files_list = QtWidgets.QListWidget(Dialog)
        self.loaded_files_list.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.loaded_files_list.setObjectName("loaded_files_list")
        self.verticalLayout.addWidget(self.loaded_files_list)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.status = QtWidgets.QLabel(Dialog)
        self.status.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.status.setText("")
        self.status.setObjectName("status")
        self.horizontalLayout_2.addWidget(self.status)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.show_lines_box = QtWidgets.QCheckBox(Dialog)
        self.show_lines_box.setObjectName("show_lines_box")
        self.horizontalLayout.addWidget(self.show_lines_box)
        self.show_labels_box = QtWidgets.QCheckBox(Dialog)
        self.show_labels_box.setObjectName("show_labels_box")
        self.horizontalLayout.addWidget(self.show_labels_box)
        self.load_file_button = QtWidgets.QPushButton(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.load_file_button.sizePolicy().hasHeightForWidth())
        self.load_file_button.setSizePolicy(sizePolicy)
        self.load_file_button.setObjectName("load_file_button")
        self.horizontalLayout.addWidget(self.load_file_button)
        self.clear_lists_button = QtWidgets.QPushButton(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.clear_lists_button.sizePolicy().hasHeightForWidth())
        self.clear_lists_button.setSizePolicy(sizePolicy)
        self.clear_lists_button.setObjectName("clear_lists_button")
        self.horizontalLayout.addWidget(self.clear_lists_button)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Reference Data"))
        self.label.setText(_translate("Dialog", "Spectral Line Lists"))
        self.show_lines_box.setText(_translate("Dialog", "Show lines"))
        self.show_labels_box.setText(_translate("Dialog", "Show labels"))
        self.load_file_button.setText(_translate("Dialog", "Load List"))
        self.clear_lists_button.setText(_translate("Dialog", "Clear Lists"))

