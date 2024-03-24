#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include "car.hpp"

int main(int argc, char *argv[])
{

    QGuiApplication app(argc, argv);
    QQmlApplicationEngine engine;

    Car car;
    engine.rootContext()->setContextProperty("carObject", &car);  // Ustaw obiekt Car w kontekście QML

    const QUrl url(QStringLiteral("Main.qml"));
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated,
                        &app, [url](QObject *obj, const QUrl &objUrl){
                            if (!obj && url == objUrl)
                                QCoreApplication::exit(-1);
                        }, Qt::QueuedConnection);
    engine.load(url);

    /*
    QObject::connect(
        &engine,
        &QQmlApplicationEngine::objectCreationFailed,
        &app,
        []() { QCoreApplication::exit(-1); },
        Qt::QueuedConnection);
    */

    return app.exec();
}
