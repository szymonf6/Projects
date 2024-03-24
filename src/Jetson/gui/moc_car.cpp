/****************************************************************************
** Meta object code from reading C++ file 'car.hpp'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.8)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "car.hpp"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'car.hpp' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.8. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_Car_t {
    QByteArrayData data[20];
    char stringdata0[222];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_Car_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_Car_t qt_meta_stringdata_Car = {
    {
QT_MOC_LITERAL(0, 0, 3), // "Car"
QT_MOC_LITERAL(1, 4, 12), // "speedChanged"
QT_MOC_LITERAL(2, 17, 0), // ""
QT_MOC_LITERAL(3, 18, 5), // "speed"
QT_MOC_LITERAL(4, 24, 19), // "batteryLevelChanged"
QT_MOC_LITERAL(5, 44, 12), // "batteryLevel"
QT_MOC_LITERAL(6, 57, 15), // "fullDistChanged"
QT_MOC_LITERAL(7, 73, 8), // "fullDist"
QT_MOC_LITERAL(8, 82, 15), // "avgSpeedChanged"
QT_MOC_LITERAL(9, 98, 8), // "avgSpeed"
QT_MOC_LITERAL(10, 107, 16), // "isWarningChanged"
QT_MOC_LITERAL(11, 124, 13), // "isLeftChanged"
QT_MOC_LITERAL(12, 138, 6), // "isLeft"
QT_MOC_LITERAL(13, 145, 14), // "isRightChanged"
QT_MOC_LITERAL(14, 160, 7), // "isRight"
QT_MOC_LITERAL(15, 168, 12), // "isPosChanged"
QT_MOC_LITERAL(16, 181, 5), // "isPos"
QT_MOC_LITERAL(17, 187, 15), // "leftDistChanged"
QT_MOC_LITERAL(18, 203, 8), // "leftDist"
QT_MOC_LITERAL(19, 212, 9) // "isWarning"

    },
    "Car\0speedChanged\0\0speed\0batteryLevelChanged\0"
    "batteryLevel\0fullDistChanged\0fullDist\0"
    "avgSpeedChanged\0avgSpeed\0isWarningChanged\0"
    "isLeftChanged\0isLeft\0isRightChanged\0"
    "isRight\0isPosChanged\0isPos\0leftDistChanged\0"
    "leftDist\0isWarning"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_Car[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       9,   14, // methods
       9,   86, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       9,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,   59,    2, 0x06 /* Public */,
       4,    1,   62,    2, 0x06 /* Public */,
       6,    1,   65,    2, 0x06 /* Public */,
       8,    1,   68,    2, 0x06 /* Public */,
      10,    1,   71,    2, 0x06 /* Public */,
      11,    1,   74,    2, 0x06 /* Public */,
      13,    1,   77,    2, 0x06 /* Public */,
      15,    1,   80,    2, 0x06 /* Public */,
      17,    1,   83,    2, 0x06 /* Public */,

 // signals: parameters
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    5,
    QMetaType::Void, QMetaType::Int,    7,
    QMetaType::Void, QMetaType::Int,    9,
    QMetaType::Void, QMetaType::Bool,   10,
    QMetaType::Void, QMetaType::Bool,   12,
    QMetaType::Void, QMetaType::Bool,   14,
    QMetaType::Void, QMetaType::Bool,   16,
    QMetaType::Void, QMetaType::Int,   18,

 // properties: name, type, flags
       3, QMetaType::Int, 0x00495001,
       5, QMetaType::Int, 0x00495001,
       7, QMetaType::Int, 0x00495001,
       9, QMetaType::Int, 0x00495001,
      18, QMetaType::Int, 0x00495001,
      19, QMetaType::Bool, 0x00495001,
      14, QMetaType::Bool, 0x00495001,
      12, QMetaType::Bool, 0x00495001,
      16, QMetaType::Bool, 0x00495001,

 // properties: notify_signal_id
       0,
       1,
       2,
       3,
       8,
       4,
       6,
       5,
       7,

       0        // eod
};

void Car::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<Car *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->speedChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: _t->batteryLevelChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 2: _t->fullDistChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 3: _t->avgSpeedChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 4: _t->isWarningChanged((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 5: _t->isLeftChanged((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 6: _t->isRightChanged((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 7: _t->isPosChanged((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 8: _t->leftDistChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (Car::*)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Car::speedChanged)) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (Car::*)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Car::batteryLevelChanged)) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (Car::*)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Car::fullDistChanged)) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (Car::*)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Car::avgSpeedChanged)) {
                *result = 3;
                return;
            }
        }
        {
            using _t = void (Car::*)(bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Car::isWarningChanged)) {
                *result = 4;
                return;
            }
        }
        {
            using _t = void (Car::*)(bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Car::isLeftChanged)) {
                *result = 5;
                return;
            }
        }
        {
            using _t = void (Car::*)(bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Car::isRightChanged)) {
                *result = 6;
                return;
            }
        }
        {
            using _t = void (Car::*)(bool );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Car::isPosChanged)) {
                *result = 7;
                return;
            }
        }
        {
            using _t = void (Car::*)(int );
            if (*reinterpret_cast<_t *>(_a[1]) == static_cast<_t>(&Car::leftDistChanged)) {
                *result = 8;
                return;
            }
        }
    }
#ifndef QT_NO_PROPERTIES
    else if (_c == QMetaObject::ReadProperty) {
        auto *_t = static_cast<Car *>(_o);
        Q_UNUSED(_t)
        void *_v = _a[0];
        switch (_id) {
        case 0: *reinterpret_cast< int*>(_v) = _t->speed(); break;
        case 1: *reinterpret_cast< int*>(_v) = _t->speed(); break;
        case 2: *reinterpret_cast< int*>(_v) = _t->fullDist(); break;
        case 3: *reinterpret_cast< int*>(_v) = _t->avgSpeed(); break;
        case 4: *reinterpret_cast< int*>(_v) = _t->leftDist(); break;
        case 5: *reinterpret_cast< bool*>(_v) = _t->isWarning(); break;
        case 6: *reinterpret_cast< bool*>(_v) = _t->isRight(); break;
        case 7: *reinterpret_cast< bool*>(_v) = _t->isLeft(); break;
        case 8: *reinterpret_cast< bool*>(_v) = _t->speed(); break;
        default: break;
        }
    } else if (_c == QMetaObject::WriteProperty) {
    } else if (_c == QMetaObject::ResetProperty) {
    }
#endif // QT_NO_PROPERTIES
}

QT_INIT_METAOBJECT const QMetaObject Car::staticMetaObject = { {
    &QObject::staticMetaObject,
    qt_meta_stringdata_Car.data,
    qt_meta_data_Car,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *Car::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *Car::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_Car.stringdata0))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int Car::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 9)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 9;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 9)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 9;
    }
#ifndef QT_NO_PROPERTIES
    else if (_c == QMetaObject::ReadProperty || _c == QMetaObject::WriteProperty
            || _c == QMetaObject::ResetProperty || _c == QMetaObject::RegisterPropertyMetaType) {
        qt_static_metacall(this, _c, _id, _a);
        _id -= 9;
    } else if (_c == QMetaObject::QueryPropertyDesignable) {
        _id -= 9;
    } else if (_c == QMetaObject::QueryPropertyScriptable) {
        _id -= 9;
    } else if (_c == QMetaObject::QueryPropertyStored) {
        _id -= 9;
    } else if (_c == QMetaObject::QueryPropertyEditable) {
        _id -= 9;
    } else if (_c == QMetaObject::QueryPropertyUser) {
        _id -= 9;
    }
#endif // QT_NO_PROPERTIES
    return _id;
}

// SIGNAL 0
void Car::speedChanged(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void Car::batteryLevelChanged(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void Car::fullDistChanged(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void Car::avgSpeedChanged(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}

// SIGNAL 4
void Car::isWarningChanged(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 4, _a);
}

// SIGNAL 5
void Car::isLeftChanged(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 5, _a);
}

// SIGNAL 6
void Car::isRightChanged(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 6, _a);
}

// SIGNAL 7
void Car::isPosChanged(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 7, _a);
}

// SIGNAL 8
void Car::leftDistChanged(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 8, _a);
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
