import math

class MathUtils:
    @staticmethod
    def floor_sqrt(n: int) -> int:
        """计算n的平方根并向下取整"""
        return int(math.floor(math.sqrt(n)))