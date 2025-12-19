import uiautomator2 as u2
import time
import cv2
import numpy as np
import random

def thoat_vao_lai_game(device: u2.Device):
    # click menu thoát game
    device.click(50, 250)
    custom_sleep(1)
    device.click(250, 390)
    custom_sleep(1)
    wait_and_click(device, "tai_khoan.png", timeout=20)
    wait_and_click(device, "tai_khoan_on.png", timeout=5)
    wait_and_click(device, "icon_game.png", timeout=5)
    wait_and_click(device, "check_log_game.png", timeout=30)
    for _ in range(3):
        custom_sleep(0.5)
        check_log_game(device)

def game_on(device: u2.Device):
    for _ in range(5):
        device.press("back")
        custom_sleep(0.5)
        x_close, y_close = find_img_center(device, "close_game.png")
        if x_close is not None and y_close is not None:
            device.click(x_close, y_close)
            custom_sleep(0.1)
            break

def check_close(device: u2.Device):
    x_close, y_close = find_img_center(device, "x.png")
    if x_close is not None and y_close is not None:
        device.click(x_close, y_close)
        custom_sleep(0.5)
        for j in range(2):
            device.click(900, 30)
            custom_sleep(0.2)
        return True
    return False


def check_log_game(device: u2.Device):
    x_log, y_log = find_img_center(device, "check_log_game.png", 0.9)
    if x_log is not None and y_log is not None:
        device.click(x_log, y_log)
        custom_sleep(0.5)
        return True
    return False


def check_lv_up(device: u2.Device):
    x_lv, y_lv = find_img_center(device, "lv_up.png")
    if x_lv is not None and y_lv is not None:
        device.click(500, 730)
        custom_sleep(0.5)
        return True
    return False
