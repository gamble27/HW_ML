import time


class Benchmark:
    """
    a custom benchmark class
    with runtime getter method
    """
    def __init__(self, function):
        self.__f = function
        self.__runtime = 0

    def __call__(self, *args, **kwargs):
        start = time.time()

        values = self.__f(*args, **kwargs)

        end = time.time()
        self.__runtime = end - start

        return values

    def get_runtime(self):
        """
        runtime getter
        :return: function runtime
        """
        return self.__runtime
