# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'cursor_location.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(812, 103)
        Dialog.setSizeGripEnabled(True)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.table_widget = QtWidgets.QTableWidget(Dialog)
        self.table_widget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.table_widget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.table_widget.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.table_widget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table_widget.setObjectName("table_widget")
        self.table_widget.setColumnCount(9)
        self.table_widget.setRowCount(1)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setHorizontalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setHorizontalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setHorizontalHeaderItem(7, item)
        item = QtWidgets.QTableWidgetItem()
        self.table_widget.setHorizontalHeaderItem(8, item)
        self.table_widget.horizontalHeader().setVisible(True)
        self.table_widget.horizontalHeader().setCascadingSectionResizes(True)
        self.table_widget.horizontalHeader().setStretchLastSection(True)
        self.table_widget.verticalHeader().setCascadingSectionResizes(True)
        self.gridLayout.addWidget(self.table_widget, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Cursor Location"))
        item = self.table_widget.verticalHeaderItem(0)
        item.setText(_translate("Dialog", "FITS filename"))
        item = self.table_widget.horizontalHeaderItem(0)
        item.setText(_translate("Dialog", "Order"))
        item = self.table_widget.horizontalHeaderItem(1)
        item.setText(_translate("Dialog", "Color"))
        item = self.table_widget.horizontalHeaderItem(2)
        item.setText(_translate("Dialog", "X Field"))
        item = self.table_widget.horizontalHeaderItem(3)
        item.setText(_translate("Dialog", "Y Field"))
        item = self.table_widget.horizontalHeaderItem(4)
        item.setText(_translate("Dialog", "X Cursor"))
        item = self.table_widget.horizontalHeaderItem(5)
        item.setText(_translate("Dialog", "Y Cursor"))
        item = self.table_widget.horizontalHeaderItem(6)
        item.setText(_translate("Dialog", "X Value"))
        item = self.table_widget.horizontalHeaderItem(7)
        item.setText(_translate("Dialog", "Y Value"))
        item = self.table_widget.horizontalHeaderItem(8)
        item.setText(_translate("Dialog", "Column"))

