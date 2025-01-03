
import logging
import colorlog


def get_logger(level=logging.INFO, log_file=None):
    # 创建logger对象
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # 创建控制台日志处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 定义颜色输出格式
    color_formatter = colorlog.ColoredFormatter(
        '%(log_color)s[%(asctime)s][%(levelname)s] %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    # 将颜色输出格式添加到控制台日志处理器
    console_handler.setFormatter(color_formatter)
    
    # 移除默认的handler
    for handler in logger.handlers:
        logger.removeHandler(handler)
    
    # 将控制台日志处理器添加到logger对象
    logger.addHandler(console_handler)
    
    # 如果提供了日志文件路径，添加文件日志处理器
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = color_formatter
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


if __name__ == '__main__':
    logger = get_logger(logging.DEBUG)
    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warning message')
    logger.error('error message')
    logger.critical('critical message')