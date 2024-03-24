import QtQuick 2.12
import QtQuick.Window 2.12

Window {
    id: root
    width: 1280
    height: 720
    visible: true
    title: qsTr("hui")

    Rectangle
    {
        id:rect
        anchors.fill: parent

        Image{
            source: "qrc:/assets/gui.png"
            anchors.fill: parent
        }

        Text{
            id: totalDistance
            text: carObject.fullDist
            color: "white"
            font.weight: Font.ExtraLight
            font.pointSize: 28
            anchors{
                top: rect.top
                topMargin: rect.height * 0.31
                right: rect.right
                rightMargin: rect.width * 0.08
            }
        }

        Text{
            id: avgSpeed
            text: carObject.avgSpeed
            color: "white"
            font.weight: Font.ExtraLight
            font.pointSize: 28
            anchors{
                top: rect.top
                topMargin: rect.height * 0.55
                right: rect.right
                rightMargin: rect.width * 0.08
            }
        }

        Text{
            id: currentSpeed
            text: carObject.speed
            color: "white"
            font.weight: Font.Light
            font.pointSize: 96
            anchors{
                top: rect.top
                topMargin: rect.height * 0.37
                right: rect.right
                rightMargin: rect.width * 0.448
            }
        }

        Text{
            id: batteryPercent
            text: carObject.batteryLevel
            color: "white"
            font.weight: Font.Light
            font.pointSize: 24
            anchors{
                top: rect.top
                topMargin: rect.height * 0.40
                right: rect.right
                rightMargin: rect.width * 0.835
            }
        }

        Text{
            id: leftDistance
            text: carObject.leftDist
            color: "white"
            font.weight: Font.Light
            font.pointSize: 18
            anchors{
                top: rect.top
                topMargin: rect.height * 0.480
                right: rect.right
                rightMargin: rect.width * 0.820
            }
        }

        Image {
            id: rightDir
            source: "qrc:/assets/right.png"
            width: rect.width * 0.03
            height: rect.height * 0.05
            visible: carObject.isRight === 1

            anchors{
                top: rect.top
                topMargin: rect.height * 0.37
                right: rect.right
                rightMargin: rect.width * 0.458

            }
        }

        Image {
            id: leftDir
            source: "qrc:/assets/left.png"
            width: rect.width * 0.03
            height: rect.height * 0.05
            visible: carObject.isLeft === 1
            anchors{
                top: rect.top
                topMargin: rect.height * 0.37
                right: rect.right
                rightMargin: rect.width * 0.51

            }
        }

        /*
        Row{
            spacing: 20
            anchors{
                top: rect.top
                topMargin: rect.height * 0.05
                right: rect.right
                rightMargin: rect.width * 0.448
            }

            Image {
                source: "/assets/mijania.png"
                width: rect.width * 0.1
                height: rect.height * 0.1
            }

            Image {
                source: "/assets/warning.png"
                width: rect.width * 0.1
                height: rect.height * 0.1
            }

            Image {
                source: "/assets/back.png"
                width: rect.width * 0.1
                height: rect.height * 0.1
            }
        }
        */
    }
}
