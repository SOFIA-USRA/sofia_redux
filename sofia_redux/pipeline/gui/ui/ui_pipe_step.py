# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'pipe_step.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(317, 34)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pipeStepFrame = QtWidgets.QFrame(Form)
        self.pipeStepFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.pipeStepFrame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.pipeStepFrame.setObjectName("pipeStepFrame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.pipeStepFrame)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.indexLabel = QtWidgets.QLabel(self.pipeStepFrame)
        self.indexLabel.setObjectName("indexLabel")
        self.horizontalLayout.addWidget(self.indexLabel)
        self.pipeStepLabel = QtWidgets.QLabel(self.pipeStepFrame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pipeStepLabel.sizePolicy().hasHeightForWidth())
        self.pipeStepLabel.setSizePolicy(sizePolicy)
        self.pipeStepLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.pipeStepLabel.setObjectName("pipeStepLabel")
        self.horizontalLayout.addWidget(self.pipeStepLabel)
        self.editButton = QtWidgets.QPushButton(self.pipeStepFrame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.editButton.sizePolicy().hasHeightForWidth())
        self.editButton.setSizePolicy(sizePolicy)
        self.editButton.setObjectName("editButton")
        self.horizontalLayout.addWidget(self.editButton)
        self.runButton = QtWidgets.QPushButton(self.pipeStepFrame)
        self.runButton.setEnabled(False)
        self.runButton.setObjectName("runButton")
        self.horizontalLayout.addWidget(self.runButton)
        self.verticalLayout.addWidget(self.pipeStepFrame)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.indexLabel.setText(_translate("Form", "1."))
        self.pipeStepLabel.setText(_translate("Form", "Pipe Step"))
        self.editButton.setText(_translate("Form", "Edit"))
        self.runButton.setText(_translate("Form", "Run"))

