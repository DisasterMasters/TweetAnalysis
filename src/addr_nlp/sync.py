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
        self.mut = threading.RLock()
        self.rcond = threading.Condition(self.mut)
        self.wcond = threading.Condition(self.mut)

    def acquire_read(self):
        """Acquire a read lock. Several threads can hold this typeof lock.
It is exclusive with write locks."""
        self.mut.acquire()
        while self.readers < 0 or self.writers:
            self.rcond.wait()
        self.readers += 1
        self.mut.release()

    def acquire_write(self):
        """Acquire a write lock. Only one thread can hold this lock, and
only when no read locks are also held."""
        self.mut.acquire()
        while self.readers != 0:
            self.writers += 1
            self.wcond.wait()
            self.writers -= 1
        self.readers = -1
        self.mut.release()

    def promote(self):
        """Promote an already-acquired read lock to a write lock
        WARNING: it is very easy to deadlock with this method"""
        self.mut.acquire()
        self.readers -= 1
        while self.readers != 0:
            self.writers += 1
            self.wcond.wait()
            self.writers -= 1
        self.readers = -1
        self.mut.release()

    def demote(self):
        """Demote an already-acquired write lock to a read lock"""
        self.mut.acquire()
        self.readers = 1
        self.rcond.notify_all()
        self.mut.release()

    def release(self):
        """Release a lock, whether read or write."""
        self.mut.acquire()
        if self.readers < 0:
            self.readers = 0
        else:
            self.readers -= 1
        wake_writers = self.writers and self.readers == 0
        wake_readers = self.writers == 0
        self.mut.release()
        if wake_writers:
            self.wcond.acquire()
            self.wcond.notify()
            self.wcond.release()
        elif wake_readers:
            self.rcond.acquire()
            self.rcond.notify_all()
            self.rcond.release()

class Channel:
    def __init__(self):
        self.mmut = threading.RLock()
        self.rmut = threading.Lock()
        self.wmut = threading.Lock()

        self.rcond = threading.Condition(self.mmut)
        self.wcond = threading.Condition(self.mmut)

        self.readers = 0
        self.writers = 0
        self.closed = False
        self.msg = None

    def __bool__(self):
        with self.mmut:
            return not self.closed

    def close(self):
        with self.mmut:
            if not self.closed:
                self.closed = True
                self.rcond.notify_all()
                self.wcond.notify_all()

    def send(self, msg):
        with self.wmut, self.mmut:
            if self.closed:
                return False

            self.msg = msg
            self.writers += 1

            if self.readers > 0:
                self.rcond.notify()

            self.wcond.wait()

            return True

    def recv(self):
        with self.rmut, self.mmut:
            while not self.closed and self.writers == 0:
                self.readers += 1
                self.rcond.wait()
                self.readers -= 1

            if self.closed:
                self.mmut.release()
                self.rmut.release()
                return None

            if self.msg is not None:
                msg = self.msg
                self.msg = None

            self.writers -= 1

            self.wcond.notify()

            self.mmut.release()
            self.rmut.release()

            return msg
