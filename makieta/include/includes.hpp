#ifndef INCLUDES_HPP_
#define INCLUDES_HPP_

/**
 * @brief plik tylko do includow zeby nie wpieprzac kilkanascie linii wszedzie tylko tu
*/
#include <unistd.h>
//zawiera definicje struktur zwiazanych z socketami
#include <sys/socket.h>
//zawiera definicje zwiazane z interfejsem sieciowym
#include <net/if.h>
//zawiera definicje zwiazane z wartowa protokołu CAN
#include <linux/can.h>
//zawiera definicje definicje dla surowego interfejsu CAN
#include <linux/can/raw.h>
//do ioctl
#include <sys/ioctl.h>

#endif /*INCLUDES_HPP_*/