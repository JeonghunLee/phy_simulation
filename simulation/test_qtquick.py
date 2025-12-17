import os
import sys
import PySide6
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

# 경로 설정
pyside_dir = os.path.dirname(PySide6.__file__)
os.environ["QT_PLUGIN_PATH"] = os.path.join(pyside_dir, "plugins")
os.environ["QML2_IMPORT_PATH"] = os.path.join(pyside_dir, "qml")
os.environ["QT_OPENGL"] = "software"

app = QGuiApplication(sys.argv)
engine = QQmlApplicationEngine()

# QML 소스 수정: Window 추가 및 visible 설정
qml_data = b"""
import QtQuick 2.15
import QtQuick.Window 2.15

Window {
    width: 400
    height: 300
    visible: true
    title: "PySide6 QML Test"

    Rectangle {
        anchors.fill: parent
        color: "steelblue"
        Text {
            anchors.centerIn: parent
            text: "Hello PySide6!"
            color: "white"
        }
    }
}
"""

engine.loadData(qml_data)

# 로드 실패 시 에러 출력 (디버깅용)
if not engine.rootObjects():
    sys.exit(-1)

sys.exit(app.exec())