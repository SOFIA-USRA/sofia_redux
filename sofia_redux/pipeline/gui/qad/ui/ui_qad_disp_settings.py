# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'qad_disp_settings.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_DisplayDialog(object):
    def setupUi(self, DisplayDialog):
        DisplayDialog.setObjectName("DisplayDialog")
        DisplayDialog.setWindowModality(QtCore.Qt.ApplicationModal)
        DisplayDialog.resize(512, 329)
        DisplayDialog.setModal(True)
        self.verticalLayout = QtWidgets.QVBoxLayout(DisplayDialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.disableDS9Box = QtWidgets.QCheckBox(DisplayDialog)
        self.disableDS9Box.setText("")
        self.disableDS9Box.setObjectName("disableDS9Box")
        self.horizontalLayout_4.addWidget(self.disableDS9Box)
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.label_9 = QtWidgets.QLabel(DisplayDialog)
        self.label_9.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_4.addWidget(self.label_9)
        self.disableEyeBox = QtWidgets.QCheckBox(DisplayDialog)
        self.disableEyeBox.setText("")
        self.disableEyeBox.setObjectName("disableEyeBox")
        self.horizontalLayout_4.addWidget(self.disableEyeBox)
        spacerItem1 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem1)
        self.label_10 = QtWidgets.QLabel(DisplayDialog)
        self.label_10.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_4.addWidget(self.label_10)
        self.disableOverplotsBox = QtWidgets.QCheckBox(DisplayDialog)
        self.disableOverplotsBox.setText("")
        self.disableOverplotsBox.setObjectName("disableOverplotsBox")
        self.horizontalLayout_4.addWidget(self.disableOverplotsBox)
        self.gridLayout.addLayout(self.horizontalLayout_4, 0, 2, 1, 1)
        self.label_8 = QtWidgets.QLabel(DisplayDialog)
        self.label_8.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 0, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(DisplayDialog)
        self.label_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 7, 0, 1, 1)
        self.scaleBox = QtWidgets.QComboBox(DisplayDialog)
        self.scaleBox.setObjectName("scaleBox")
        self.scaleBox.addItem("")
        self.scaleBox.addItem("")
        self.scaleBox.addItem("")
        self.scaleBox.addItem("")
        self.gridLayout.addWidget(self.scaleBox, 6, 2, 1, 1)
        self.label_6 = QtWidgets.QLabel(DisplayDialog)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 6, 0, 1, 1, QtCore.Qt.AlignRight)
        self.lockSliceBox = QtWidgets.QComboBox(DisplayDialog)
        self.lockSliceBox.setObjectName("lockSliceBox")
        self.lockSliceBox.addItem("")
        self.lockSliceBox.addItem("")
        self.lockSliceBox.addItem("")
        self.gridLayout.addWidget(self.lockSliceBox, 5, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(DisplayDialog)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 5, 0, 1, 1, QtCore.Qt.AlignRight)
        self.lockImageBox = QtWidgets.QComboBox(DisplayDialog)
        self.lockImageBox.setObjectName("lockImageBox")
        self.lockImageBox.addItem("")
        self.lockImageBox.addItem("")
        self.lockImageBox.addItem("")
        self.gridLayout.addWidget(self.lockImageBox, 4, 2, 1, 1)
        self.label = QtWidgets.QLabel(DisplayDialog)
        self.label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 4, 0, 1, 1)
        self.extensionBox = QtWidgets.QComboBox(DisplayDialog)
        self.extensionBox.setEditable(True)
        self.extensionBox.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.extensionBox.setObjectName("extensionBox")
        self.extensionBox.addItem("")
        self.extensionBox.addItem("")
        self.extensionBox.addItem("")
        self.extensionBox.addItem("")
        self.extensionBox.addItem("")
        self.gridLayout.addWidget(self.extensionBox, 3, 2, 1, 1)
        self.label_7 = QtWidgets.QLabel(DisplayDialog)
        self.label_7.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 3, 0, 1, 1)
        self.colorMapBox = QtWidgets.QLineEdit(DisplayDialog)
        self.colorMapBox.setObjectName("colorMapBox")
        self.gridLayout.addWidget(self.colorMapBox, 7, 2, 1, 1)
        self.label_4 = QtWidgets.QLabel(DisplayDialog)
        self.label_4.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 8, 0, 1, 1)
        self.zoomBox = QtWidgets.QCheckBox(DisplayDialog)
        self.zoomBox.setText("")
        self.zoomBox.setObjectName("zoomBox")
        self.gridLayout.addWidget(self.zoomBox, 8, 2, 1, 1, QtCore.Qt.AlignLeft)
        self.label_5 = QtWidgets.QLabel(DisplayDialog)
        self.label_5.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 9, 0, 1, 1)
        self.tileBox = QtWidgets.QCheckBox(DisplayDialog)
        self.tileBox.setText("")
        self.tileBox.setObjectName("tileBox")
        self.gridLayout.addWidget(self.tileBox, 9, 2, 1, 1)
        self.label_11 = QtWidgets.QLabel(DisplayDialog)
        self.label_11.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 10, 0, 1, 1)
        self.snRangeBox = QtWidgets.QLineEdit(DisplayDialog)
        self.snRangeBox.setObjectName("snRangeBox")
        self.gridLayout.addWidget(self.snRangeBox, 10, 2, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(DisplayDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok|QtWidgets.QDialogButtonBox.Reset|QtWidgets.QDialogButtonBox.RestoreDefaults)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(DisplayDialog)
        self.buttonBox.accepted.connect(DisplayDialog.accept)
        self.buttonBox.rejected.connect(DisplayDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(DisplayDialog)
        DisplayDialog.setTabOrder(self.extensionBox, self.lockImageBox)
        DisplayDialog.setTabOrder(self.lockImageBox, self.lockSliceBox)
        DisplayDialog.setTabOrder(self.lockSliceBox, self.scaleBox)
        DisplayDialog.setTabOrder(self.scaleBox, self.colorMapBox)
        DisplayDialog.setTabOrder(self.colorMapBox, self.zoomBox)
        DisplayDialog.setTabOrder(self.zoomBox, self.tileBox)

    def retranslateUi(self, DisplayDialog):
        _translate = QtCore.QCoreApplication.translate
        DisplayDialog.setWindowTitle(_translate("DisplayDialog", "Display Settings"))
        self.disableDS9Box.setToolTip(_translate("DisplayDialog", "Set to disable DS9 display."))
        self.label_9.setText(_translate("DisplayDialog", "Disable Eye"))
        self.disableEyeBox.setToolTip(_translate("DisplayDialog", "Set to disable spectral display."))
        self.label_10.setText(_translate("DisplayDialog", "Disable overplots"))
        self.disableOverplotsBox.setToolTip(_translate("DisplayDialog", "Set to disable automated overplots in DS9."))
        self.label_8.setText(_translate("DisplayDialog", "Disable DS9"))
        self.label_2.setText(_translate("DisplayDialog", "Color map"))
        self.scaleBox.setToolTip(_translate("DisplayDialog", "If none, scale will not be updated on load."))
        self.scaleBox.setItemText(0, _translate("DisplayDialog", "ZScale"))
        self.scaleBox.setItemText(1, _translate("DisplayDialog", "MinMax"))
        self.scaleBox.setItemText(2, _translate("DisplayDialog", "ZMax"))
        self.scaleBox.setItemText(3, _translate("DisplayDialog", "none"))
        self.label_6.setText(_translate("DisplayDialog", "Scale"))
        self.lockSliceBox.setToolTip(_translate("DisplayDialog", "If set to WCS, cube slices are locked to calibrated coordinates."))
        self.lockSliceBox.setItemText(0, _translate("DisplayDialog", "WCS"))
        self.lockSliceBox.setItemText(1, _translate("DisplayDialog", "Image"))
        self.lockSliceBox.setItemText(2, _translate("DisplayDialog", "None"))
        self.label_3.setText(_translate("DisplayDialog", "Lock slice to"))
        self.lockImageBox.setToolTip(_translate("DisplayDialog", "If set to WCS, frames and crosshair will be locked in sky coordinates."))
        self.lockImageBox.setCurrentText(_translate("DisplayDialog", "WCS"))
        self.lockImageBox.setItemText(0, _translate("DisplayDialog", "WCS"))
        self.lockImageBox.setItemText(1, _translate("DisplayDialog", "Image"))
        self.lockImageBox.setItemText(2, _translate("DisplayDialog", "None"))
        self.label.setText(_translate("DisplayDialog", "Lock frames to"))
        self.extensionBox.setToolTip(_translate("DisplayDialog", "Select extension display strategy, or edit to specify an extension number or name."))
        self.extensionBox.setCurrentText(_translate("DisplayDialog", "First"))
        self.extensionBox.setItemText(0, _translate("DisplayDialog", "First"))
        self.extensionBox.setItemText(1, _translate("DisplayDialog", "All (in separate frames)"))
        self.extensionBox.setItemText(2, _translate("DisplayDialog", "All (in a cube)"))
        self.extensionBox.setItemText(3, _translate("DisplayDialog", "S/N"))
        self.extensionBox.setItemText(4, _translate("DisplayDialog", "EXTNUM (edit to select a specific extension)"))
        self.label_7.setText(_translate("DisplayDialog", "Extension to display"))
        self.colorMapBox.setToolTip(_translate("DisplayDialog", "If none, colormap will not be updated on load."))
        self.label_4.setText(_translate("DisplayDialog", "Zoom to fit"))
        self.zoomBox.setToolTip(_translate("DisplayDialog", "Set to zoom to fit the last loaded frame."))
        self.label_5.setText(_translate("DisplayDialog", "Tile images"))
        self.tileBox.setToolTip(_translate("DisplayDialog", "Set to tile all loaded images in the DS9 window."))
        self.label_11.setText(_translate("DisplayDialog", "S/N range"))
        self.snRangeBox.setToolTip(_translate("DisplayDialog", "Set range for S/N image as min,max if desired."))

