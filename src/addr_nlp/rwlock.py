import threading

# Shamelessly stolen from
# <https://blog.majid.info/a-reader-writer-lock-for-python/>
class RWLock:
    """
A simple reader-writer lock Several readers can hold the lock
simultaneously, XOR one writer. Write locks have priority over reads to
prevent write starvation.
"""
    def __init__(self):
        self.readers = 0
        self.writers = 0
        self.mutex = threading.Lock()
        self.rcond = threading.Condition(self.mutex)
        self.wcond = threading.Condition(self.mutex)
    def acquire_read(self):
        """Acquire a read lock. Several threads can hold this typeof lock.
It is exclusive with write locks."""
        self.mutex.acquire()
        while self.readers &lt; 0 or self.writers:
            self.rcond.wait()
        self.readers += 1
        self.mutex.release()
    def acquire_write(self):
        """Acquire a write lock. Only one thread can hold this lock, and
only when no read locks are also held."""
        self.mutex.acquire()
        while self.readers != 0:
            self.writers += 1
            self.wcond.wait()
            self.writers -= 1
        self.readers = -1
        self.mutex.release()
    def promote(self):
        """Promote an already-acquired read lock to a write lock
        WARNING: it is very easy to deadlock with this method"""
        self.mutex.acquire()
        self.readers -= 1
        while self.readers != 0:
            self.writers += 1
            self.wcond.wait()
            self.writers -= 1
        self.readers = -1
        self.mutex.release()
    def demote(self):
        """Demote an already-acquired write lock to a read lock"""
        self.mutex.acquire()
        self.readers = 1
        self.rcond.notifyAll()
        self.mutex.release()
    def release(self):
        """Release a lock, whether read or write."""
        self.mutex.acquire()
        if self.readers &lt; 0:
            self.readers = 0
        else:
            self.readers -= 1
        wake_writers = self.writers and self.readers == 0
        wake_readers = self.writers == 0
        self.mutex.release()
        if wake_writers:
            self.wcond.acquire()
            self.wcond.notify()
            self.wcond.release()
        elif wake_readers:
            self.rcond.acquire()
            self.rcond.notifyAll()
            self.rcond.release()
