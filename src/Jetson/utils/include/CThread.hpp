#ifndef __THREAD_HPP_
#define __THREAD_HPP_

#include <pthread.h>
#include "ReturnCode.hpp"
#include <unistd.h>
#include <cstdint>

/**
* Wskaźnik funkcji typedef reprezentujący główny punkt wejścia wątku.
 * Parametr zdefiniowany przez użytkownika jest przekazywany w taki sposób, że użytkownik może to zrobić
 * przekazując dane lub inną wartość do swojego wątku, aby mógł się on sam zidentyfikować.
 * @typedef threads
 */
typedef void* (*ThreadEntryFunction)( void* user_param );


/**
* Aby utworzyć własny wątek, podaj wskaźnik funkcji punktu wejścia wątku,
 * lub dziedzicz z tej klasy i zaimplementuj własną funkcję Run().
 * @class Thread
 */
class Thread
{
public:
	/**
	 * konstruktor
	 */
	Thread();

	/**
	 * Destruktor automatycznie stopuje wątek
	 */
	~Thread();

	/**
	 * @brief Startuje wątek jak ją wywołasz
	 */
	void run();

	/**
	 * @brief start wątku.
     * @fn Start()
	 * @result zwraca błąd jeżeli nie udał się start wątku.
	 */
	bool Start();

	/**
	 * @brief Rozpocznij wątek, korzystając ze wskaźnika funkcji wejściowej dostarczonego przez użytkownika.
     * @fn Start( ThreadEntryFunction entry, void* user_param=NULL )
	 * @result zwraca błąd jeżeli nie udał się start wątku.
	 */
	bool Start( ThreadEntryFunction entry, void* user_param=NULL );
	
	/**
	 * @brief zatrzymaj wątek
     * @fn Stop(bool wait = false)
	 * @result jeżeli jest true to bedzie zablokowane Stop() dopoki thread sie nie zakonczył
	 */
	ReturnCode Stop(bool wait=false);
	
	/**
	 * @brief uzyskuje najwyższy numer priorytetu dostępny
     * @fn GetMaxPriority()
	 */
	static int GetMaxPriority();
	
	/**
	 * @brief uzyskuje najniższy dostępny numer priorytetu
     * @fn GetMinPriority()
	 */
	static int GetMinPriority();

	/**
	 * @brief uzyskuje numer priorytu wątku
     * @fn GetPriority( pthread_t* thread=NULL)
	 */
	static int GetPriority( pthread_t* thread=NULL );
 
	/**
	 * @brief nadaj wątkowi priorytet
     * @fn SetPriority( int priority, pthread_t* thread=NULL )
	 */
	static int SetPriority( int priority, pthread_t* thread=NULL );

	/**
	 * uzyskaj priorytet wątka
	 */
	int GetPriorityLevel();

	/**
	 * nadaj wątkowi priorytet
	 */
	bool SetPriorityLevel( int priority );

	/**
     * @brief Niezależnie od wątku, z którego wywołujesz, daj procesorowi określoną liczbę milisekund.
     * @fn Yield(uint32_t ms)
	 */
	ReturnCode Yield( uint32_t ms );

	/**
     * @brief Get thread identififer
     * @fn GetThreadID()
	 */
	inline pthread_t* GetThreadID() { return &mThreadID; }

	/**
	 * @brief Look up which CPU core the thread is running on.
     * @fn GetCPU();
	 */
	static int GetCPU();

private:
    /**
     * @brief służy do zablokowania dostępu do sekcji przy użyciu mutexa.
     * Po wywołaniu tej metody, operacje na współdzielonych danych są chronione przed równoczesnym dostępem wielu wątków.
     * @fn Lock()
    */
    void Lock() { pthread_mutex_lock(&mMutex); }

    /**
     * służy do odblokowania dostępu do sekcji krytycznej, wcześniej zablokowanej przez metodę Lock().
     * @fn Unlock()
     */ 
    void Unlock() { pthread_mutex_unlock(&mMutex); }

    /**
     * @brief mutex używany do synchronizacji dostępu do współdzielonych danych pomiędzy wątkami.
     * @fn mMutex
    */
    pthread_mutex_t mMutex;

    /**
     * @brief domyślna funkcja, która jest wywoływana po utworzeniu nowego wątku.
     * Ta funkcja jest odpowiedzialna za wywołanie metody Run() na obiekcie wątku.
     * @fn DefaultEntry
    */
    ReturnCode DefaultEntry(void* param);

    /**
     * @brief ID wątku, przypisane po jego utworzeniu.
     * @fn mThreadID
    */
    pthread_t mThreadID;

    /**
     * @brief Flaga informująca, czy wątek został uruchomiony.
     * @fn mThreadStarted
    */
    bool mThreadStarted;
};

#endif /*__THREAD_HPP_*/