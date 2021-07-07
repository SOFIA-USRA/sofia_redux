# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'textview.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_TextWindow(object):
    def setupUi(self, TextWindow):
        TextWindow.setObjectName("TextWindow")
        TextWindow.resize(587, 629)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(TextWindow.sizePolicy().hasHeightForWidth())
        TextWindow.setSizePolicy(sizePolicy)
        self.gridLayout = QtWidgets.QGridLayout(TextWindow)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.findText = QtWidgets.QLineEdit(TextWindow)
        self.findText.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.findText.setObjectName("findText")
        self.horizontalLayout.addWidget(self.findText)
        self.findButton = QtWidgets.QPushButton(TextWindow)
        self.findButton.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.findButton.setObjectName("findButton")
        self.horizontalLayout.addWidget(self.findButton)
        self.filterButton = QtWidgets.QPushButton(TextWindow)
        self.filterButton.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.filterButton.setObjectName("filterButton")
        self.horizontalLayout.addWidget(self.filterButton)
        self.tableButton = QtWidgets.QPushButton(TextWindow)
        self.tableButton.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.tableButton.setObjectName("tableButton")
        self.horizontalLayout.addWidget(self.tableButton)
        self.saveButton = QtWidgets.QPushButton(TextWindow)
        self.saveButton.setEnabled(True)
        self.saveButton.setObjectName("saveButton")
        self.horizontalLayout.addWidget(self.saveButton)
        self.gridLayout.addLayout(self.horizontalLayout, 2, 0, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.textEdit = QtWidgets.QTextEdit(TextWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.textEdit.sizePolicy().hasHeightForWidth())
        self.textEdit.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.textEdit.setFont(font)
        self.textEdit.setReadOnly(True)
        self.textEdit.setObjectName("textEdit")
        self.verticalLayout.addWidget(self.textEdit)
        self.gridLayout.addLayout(self.verticalLayout, 1, 0, 1, 1)

        self.retranslateUi(TextWindow)
        QtCore.QMetaObject.connectSlotsByName(TextWindow)
        TextWindow.setTabOrder(self.textEdit, self.findText)
        TextWindow.setTabOrder(self.findText, self.filterButton)
        TextWindow.setTabOrder(self.filterButton, self.tableButton)

    def retranslateUi(self, TextWindow):
        _translate = QtCore.QCoreApplication.translate
        TextWindow.setWindowTitle(_translate("TextWindow", "Dialog"))
        self.findButton.setText(_translate("TextWindow", "Find"))
        self.filterButton.setText(_translate("TextWindow", "Filter"))
        self.tableButton.setText(_translate("TextWindow", "Table"))
        self.saveButton.setText(_translate("TextWindow", "Save"))

