# -*- coding: UTF-8 -*-
from python.edgesmart.core import base_object


class base_rule(base_object):
    """
    规则父类
    """

    def __init__(self):
        pass

    def router_detection_class(self):
        """
        通过这个方法可以路由到具体的检测类
        :return:
        """
        pass


