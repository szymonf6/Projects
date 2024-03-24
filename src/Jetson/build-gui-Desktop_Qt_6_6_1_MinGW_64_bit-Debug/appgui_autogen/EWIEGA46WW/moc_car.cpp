/****************************************************************************
** Meta object code from reading C++ file 'car.hpp'
**
** Created by: The Qt Meta Object Compiler version 68 (Qt 6.6.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../gui/car.hpp"
#include <QtCore/qmetatype.h>

#if __has_include(<QtCore/qtmochelpers.h>)
#include <QtCore/qtmochelpers.h>
#else
QT_BEGIN_MOC_NAMESPACE
#endif


#include <memory>

#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'car.hpp' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 68
#error "This file was generated using the moc from 6.6.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

#ifndef Q_CONSTINIT
#define Q_CONSTINIT
#endif

QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
QT_WARNING_DISABLE_GCC("-Wuseless-cast")
namespace {

#ifdef QT_MOC_HAS_STRINGDATA
struct qt_meta_stringdata_CLASSCarENDCLASS_t {};
static constexpr auto qt_meta_stringdata_CLASSCarENDCLASS = QtMocHelpers::stringData(
    "Car",
    "speedChanged",
    "",
    "speed",
    "batteryLevelChanged",
    "batteryLevel",
    "fullDistChanged",
    "fullDist",
    "avgSpeedChanged",
    "avgSpeed",
    "isWarningChanged",
    "isLeftChanged",
    "isLeft",
    "isRightChanged",
    "isRight",
    "isPosChanged",
    "isPos",
    "leftDistChanged",
    "leftDist",
    "isWarning"
);
#else  // !QT_MOC_HAS_STRING_DATA
struct qt_meta_stringdata_CLASSCarENDCLASS_t {
    uint offsetsAndSizes[40];
    char stringdata0[4];
    char stringdata1[13];
    char stringdata2[1];
    char stringdata3[6];
    char stringdata4[20];
    char stringdata5[13];
    char stringdata6[16];
    char stringdata7[9];
    char stringdata8[16];
    char stringdata9[9];
    char stringdata10[17];
    char stringdata11[14];
    char stringdata12[7];
    char stringdata13[15];
    char stringdata14[8];
    char stringdata15[13];
    char stringdata16[6];
    char stringdata17[16];
    char stringdata18[9];
    char stringdata19[10];
};
#define QT_MOC_LITERAL(ofs, len) \
    uint(sizeof(qt_meta_stringdata_CLASSCarENDCLASS_t::offsetsAndSizes) + ofs), len 
Q_CONSTINIT static const qt_meta_stringdata_CLASSCarENDCLASS_t qt_meta_stringdata_CLASSCarENDCLASS = {
    {
        QT_MOC_LITERAL(0, 3),  // "Car"
        QT_MOC_LITERAL(4, 12),  // "speedChanged"
        QT_MOC_LITERAL(17, 0),  // ""
        QT_MOC_LITERAL(18, 5),  // "speed"
        QT_MOC_LITERAL(24, 19),  // "batteryLevelChanged"
        QT_MOC_LITERAL(44, 12),  // "batteryLevel"
        QT_MOC_LITERAL(57, 15),  // "fullDistChanged"
        QT_MOC_LITERAL(73, 8),  // "fullDist"
        QT_MOC_LITERAL(82, 15),  // "avgSpeedChanged"
        QT_MOC_LITERAL(98, 8),  // "avgSpeed"
        QT_MOC_LITERAL(107, 16),  // "isWarningChanged"
        QT_MOC_LITERAL(124, 13),  // "isLeftChanged"
        QT_MOC_LITERAL(138, 6),  // "isLeft"
        QT_MOC_LITERAL(145, 14),  // "isRightChanged"
        QT_MOC_LITERAL(160, 7),  // "isRight"
        QT_MOC_LITERAL(168, 12),  // "isPosChanged"
        QT_MOC_LITERAL(181, 5),  // "isPos"
        QT_MOC_LITERAL(187, 15),  // "leftDistChanged"
        QT_MOC_LITERAL(203, 8),  // "leftDist"
        QT_MOC_LITERAL(212, 9)   // "isWarning"
    },
    "Car",
    "speedChanged",
    "",
    "speed",
    "batteryLevelChanged",
    "batteryLevel",
    "fullDistChanged",
    "fullDist",
    "avgSpeedChanged",
    "avgSpeed",
    "isWarningChanged",
    "isLeftChanged",
    "isLeft",
    "isRightChanged",
    "isRight",
    "isPosChanged",
    "isPos",
    "leftDistChanged",
    "leftDist",
    "isWarning"
};
#undef QT_MOC_LITERAL
#endif // !QT_MOC_HAS_STRING_DATA
} // unnamed namespace

Q_CONSTINIT static const uint qt_meta_data_CLASSCarENDCLASS[] = {

 // content:
      12,       // revision
       0,       // classname
       0,    0, // classinfo
       9,   14, // methods
       9,   95, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       9,       // signalCount

 // signals: name, argc, parameters, tag, flags, initial metatype offsets
       1,    1,   68,    2, 0x06,   10 /* Public */,
       4,    1,   71,    2, 0x06,   12 /* Public */,
       6,    1,   74,    2, 0x06,   14 /* Public */,
       8,    1,   77,    2, 0x06,   16 /* Public */,
      10,    1,   80,    2, 0x06,   18 /* Public */,
      11,    1,   83,    2, 0x06,   20 /* Public */,
      13,    1,   86,    2, 0x06,   22 /* Public */,
      15,    1,   89,    2, 0x06,   24 /* Public */,
      17,    1,   92,    2, 0x06,   26 /* Public */,

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
       3, QMetaType::Int, 0x00015001, uint(0), 0,
       5, QMetaType::Int, 0x00015001, uint(1), 0,
       7, QMetaType::Int, 0x00015001, uint(2), 0,
       9, QMetaType::Int, 0x00015001, uint(3), 0,
      18, QMetaType::Int, 0x00015001, uint(8), 0,
      19, QMetaType::Bool, 0x00015001, uint(4), 0,
      14, QMetaType::Bool, 0x00015001, uint(6), 0,
      12, QMetaType::Bool, 0x00015001, uint(5), 0,
      16, QMetaType::Bool, 0x00015001, uint(7), 0,

       0        // eod
};

Q_CONSTINIT const QMetaObject Car::staticMetaObject = { {
    QMetaObject::SuperData::link<QObject::staticMetaObject>(),
    qt_meta_stringdata_CLASSCarENDCLASS.offsetsAndSizes,
    qt_meta_data_CLASSCarENDCLASS,
    qt_static_metacall,
    nullptr,
    qt_incomplete_metaTypeArray<qt_meta_stringdata_CLASSCarENDCLASS_t,
        // property 'speed'
        QtPrivate::TypeAndForceComplete<int, std::true_type>,
        // property 'batteryLevel'
        QtPrivate::TypeAndForceComplete<int, std::true_type>,
        // property 'fullDist'
        QtPrivate::TypeAndForceComplete<int, std::true_type>,
        // property 'avgSpeed'
        QtPrivate::TypeAndForceComplete<int, std::true_type>,
        // property 'leftDist'
        QtPrivate::TypeAndForceComplete<int, std::true_type>,
        // property 'isWarning'
        QtPrivate::TypeAndForceComplete<bool, std::true_type>,
        // property 'isRight'
        QtPrivate::TypeAndForceComplete<bool, std::true_type>,
        // property 'isLeft'
        QtPrivate::TypeAndForceComplete<bool, std::true_type>,
        // property 'isPos'
        QtPrivate::TypeAndForceComplete<bool, std::true_type>,
        // Q_OBJECT / Q_GADGET
        QtPrivate::TypeAndForceComplete<Car, std::true_type>,
        // method 'speedChanged'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'batteryLevelChanged'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'fullDistChanged'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'avgSpeedChanged'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>,
        // method 'isWarningChanged'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        // method 'isLeftChanged'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        // method 'isRightChanged'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        // method 'isPosChanged'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<bool, std::false_type>,
        // method 'leftDistChanged'
        QtPrivate::TypeAndForceComplete<void, std::false_type>,
        QtPrivate::TypeAndForceComplete<int, std::false_type>
    >,
    nullptr
} };

void Car::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<Car *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->speedChanged((*reinterpret_cast< std::add_pointer_t<int>>(_a[1]))); break;
        case 1: _t->batteryLevelChanged((*reinterpret_cast< std::add_pointer_t<int>>(_a[1]))); break;
        case 2: _t->fullDistChanged((*reinterpret_cast< std::add_pointer_t<int>>(_a[1]))); break;
        case 3: _t->avgSpeedChanged((*reinterpret_cast< std::add_pointer_t<int>>(_a[1]))); break;
        case 4: _t->isWarningChanged((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1]))); break;
        case 5: _t->isLeftChanged((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1]))); break;
        case 6: _t->isRightChanged((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1]))); break;
        case 7: _t->isPosChanged((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1]))); break;
        case 8: _t->leftDistChanged((*reinterpret_cast< std::add_pointer_t<int>>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        {
            using _t = void (Car::*)(int );
            if (_t _q_method = &Car::speedChanged; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 0;
                return;
            }
        }
        {
            using _t = void (Car::*)(int );
            if (_t _q_method = &Car::batteryLevelChanged; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 1;
                return;
            }
        }
        {
            using _t = void (Car::*)(int );
            if (_t _q_method = &Car::fullDistChanged; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 2;
                return;
            }
        }
        {
            using _t = void (Car::*)(int );
            if (_t _q_method = &Car::avgSpeedChanged; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 3;
                return;
            }
        }
        {
            using _t = void (Car::*)(bool );
            if (_t _q_method = &Car::isWarningChanged; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 4;
                return;
            }
        }
        {
            using _t = void (Car::*)(bool );
            if (_t _q_method = &Car::isLeftChanged; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 5;
                return;
            }
        }
        {
            using _t = void (Car::*)(bool );
            if (_t _q_method = &Car::isRightChanged; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 6;
                return;
            }
        }
        {
            using _t = void (Car::*)(bool );
            if (_t _q_method = &Car::isPosChanged; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 7;
                return;
            }
        }
        {
            using _t = void (Car::*)(int );
            if (_t _q_method = &Car::leftDistChanged; *reinterpret_cast<_t *>(_a[1]) == _q_method) {
                *result = 8;
                return;
            }
        }
    } else if (_c == QMetaObject::ReadProperty) {
        auto *_t = static_cast<Car *>(_o);
        (void)_t;
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
    } else if (_c == QMetaObject::BindableProperty) {
    }
}

const QMetaObject *Car::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *Car::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_CLASSCarENDCLASS.stringdata0))
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
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 9;
    }else if (_c == QMetaObject::ReadProperty || _c == QMetaObject::WriteProperty
            || _c == QMetaObject::ResetProperty || _c == QMetaObject::BindableProperty
            || _c == QMetaObject::RegisterPropertyMetaType) {
        qt_static_metacall(this, _c, _id, _a);
        _id -= 9;
    }
    return _id;
}

// SIGNAL 0
void Car::speedChanged(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void Car::batteryLevelChanged(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void Car::fullDistChanged(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void Car::avgSpeedChanged(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}

// SIGNAL 4
void Car::isWarningChanged(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 4, _a);
}

// SIGNAL 5
void Car::isLeftChanged(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 5, _a);
}

// SIGNAL 6
void Car::isRightChanged(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 6, _a);
}

// SIGNAL 7
void Car::isPosChanged(bool _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 7, _a);
}

// SIGNAL 8
void Car::leftDistChanged(int _t1)
{
    void *_a[] = { nullptr, const_cast<void*>(reinterpret_cast<const void*>(std::addressof(_t1))) };
    QMetaObject::activate(this, &staticMetaObject, 8, _a);
}
QT_WARNING_POP
