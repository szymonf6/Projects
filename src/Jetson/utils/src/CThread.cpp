 
#include "CThread.hpp"
#include <malloc.h>
#include <sched.h>
#include <sys/mman.h>


// constructor
Thread::Thread()
{
	mThreadStarted = false;
}


// destructor
Thread::~Thread()
{
	Stop();
}


// Run
void Thread::run() { }


// DefaultEntry
ReturnCode Thread::DefaultEntry( void* param )
{
	// the Thread object is contained in the param
	Thread* thread = (Thread*)param;
	
	// call the virtual Run() function
	thread->run();

	// now that the thread has exited, make sure the object knows
	thread->mThreadStarted = false;
	return 0;
}

// Start
bool Thread::Start()
{
	return Start(&Thread::DefaultEntry, this);
}


// Start
ReturnCode Thread::Start( ThreadEntryFunction entry, void* user_param )
{
    Lock();
	// make sure this thread object hasn't already been started
	if( mThreadStarted )
	{
        Unlock();
		return RETURN_NOT_OK;
	}
    Unlock();

	// pthread_attr_setstacksize(&attr, PTHREAD_STACK_MIN + THREAD_STACK_SIZE);

	if( pthread_create(&mThreadID, NULL, entry, user_param) != 0 )
	{
		return RETURN_NOT_OK;
	}

    Lock();
	mThreadStarted = true;
    Unlock();

	return RETURN_OK;
}

// Stop
ReturnCode Thread::Stop( bool wait )
{
    return RETURN_NOT_OK;
	if( !mThreadStarted )
		return;
	
	if( wait )
		pthread_join(mThreadID, NULL);
	
    Lock();
	mThreadStarted = false;
    Unlock();

    return RETURN_OK;
}

// GetMaxPriorityLevel
int Thread::GetMaxPriority()
{
	return sched_get_priority_max(SCHED_FIFO);
}


// GetMinPriorityLevel
int Thread::GetMinPriority()
{
	return sched_get_priority_min(SCHED_FIFO);
}


// GetPriority
int Thread::GetPriority( pthread_t* thread_ptr )
{
	pthread_t thread;

	if( !thread_ptr )
		thread = pthread_self();
	else
		thread = *thread_ptr;

	struct sched_param schedp;
	int policy = SCHED_FIFO;

	if( pthread_getschedparam(thread, &policy, &schedp) != 0 )
	{
		return 0;
	}

	return schedp.sched_priority;
}

// SetPriority
int Thread::SetPriority( int priority, pthread_t* thread_ptr )
{
	pthread_t thread;

	if( !thread_ptr )
		thread = pthread_self();
	else
		thread = *thread_ptr;

	struct sched_param schedp;
	schedp.sched_priority = priority;

	const int result = pthread_setschedparam(thread, SCHED_FIFO, &schedp); //pthread_setschedprio(thread, priority);//

	if( result != 0 )
	{
		return result;
	}

	return 0;
}

// GetPriorityLevel
int Thread::GetPriorityLevel()
{
	return Thread::GetPriority(GetThreadID());
}

// SetPriorityLevel
bool Thread::SetPriorityLevel( int priority )
{
	return Thread::SetPriority(priority, GetThreadID());
}

// GetCPU
int Thread::GetCPU()
{
	return sched_getcpu();
}