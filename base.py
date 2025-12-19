import uiautomator2 as u2
import time
import cv2
import numpy as np
import random

def check_has_img(
    d: "u2.Device",  # Sử dụng string "u2.Device" để tránh lỗi import nếu không dùng
    template_path: str,
    threshold: float = 0.85,
    roi: tuple = None,  # (x1, y1, x2, y2) - khu vực cần kiểm tra
) -> bool:
    """
    Kiểm tra xem một hình ảnh mẫu có xuất hiện trên màn hình của thiết bị hay không.

    Args:
        d: Đối tượng thiết bị uiautomator2.
        template_path: Đường dẫn tương đối đến tệp hình ảnh mẫu trong thư mục "assets".
        threshold: Ngưỡng tương đồng để xác định một kết quả khớp (từ 0.0 đến 1.0).
        roi: Tuple (x1, y1, x2, y2) định nghĩa khu vực cần kiểm tra.
             Nếu None thì kiểm tra toàn màn hình.

    Returns:
        True nếu tìm thấy hình ảnh mẫu, False nếu không.
    """
    try:
        # 1. Chụp ảnh màn hình từ thiết bị
        img_rgb = d.screenshot(format="opencv")
        if img_rgb is None:
            print("Lỗi: Không thể chụp ảnh màn hình.")
            return False

        # 2. Cắt ảnh theo ROI nếu được chỉ định
        if roi is not None:
            x1, y1, x2, y2 = roi
            # Đảm bảo tọa độ nằm trong phạm vi ảnh
            height, width = img_rgb.shape[:2]
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            if x1 >= x2 or y1 >= y2:
                print("Lỗi: ROI không hợp lệ.")
                return False

            img_rgb = img_rgb[y1:y2, x1:x2]

        # 3. Đọc ảnh mẫu
        template = cv2.imread(f"assets/{template_path}")
        if template is None:
            print(f"Lỗi: Không thể đọc tệp mẫu tại: assets/{template_path}")
            return False

        # 4. Kiểm tra kích thước template có phù hợp với ROI không
        template_height, template_width = template.shape[:2]
        img_height, img_width = img_rgb.shape[:2]

        if template_height > img_height or template_width > img_width:
            print(
                f"Cảnh báo: {template_path} ({template_width}x{template_height}) lớn hơn vùng tìm kiếm ({img_width}x{img_height})"
            )
            return False

        # 5. Thực hiện so khớp mẫu
        res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)

        # 6. Tìm các vị trí có độ tương đồng cao hơn ngưỡng
        loc = np.where(res >= threshold)

        # 7. Kiểm tra kết quả
        if len(loc[0]) > 0:
            return True
        else:
            return False

    except Exception as e:
        print(f"Lỗi trong check_has_img: {str(e)}")
        return False

    except Exception as e:
        return False


def find_img_center(
    d: "u2.Device",
    template_path: str,
    threshold: float = 0.85,
) -> tuple:
    """
    Tìm hình ảnh mẫu trên màn hình và trả về tọa độ giữa của ảnh được tìm thấy.

    Args:
        d: Đối tượng thiết bị uiautomator2.
        template_path: Đường dẫn tương đối đến tệp hình ảnh mẫu trong thư mục "assets".
        threshold: Ngưỡng tương đồng để xác định một kết quả khớp (từ 0.0 đến 1.0).

    Returns:
        Tuple (x, y) là tọa độ giữa của ảnh được tìm thấy.
        Trả về (None, None) nếu không tìm thấy ảnh.
    """
    try:
        # 1. Chụp ảnh màn hình từ thiết bị
        img_rgb = d.screenshot(format="opencv")
        if img_rgb is None:
            print("Lỗi: Không thể chụp ảnh màn hình.")
            return (None, None)

        # 2. Đọc ảnh mẫu
        template = cv2.imread(f"assets/{template_path}")
        if template is None:
            print(f"Lỗi: Không thể đọc tệp mẫu tại: assets/{template_path}")
            return (None, None)

        # 3. Lấy kích thước của template
        template_height, template_width = template.shape[:2]

        # 4. Thực hiện so khớp mẫu
        res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)

        # 5. Tìm vị trí có độ tương đồng cao nhất
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # 6. Kiểm tra nếu độ tương đồng cao nhất >= threshold
        if max_val >= threshold:
            # Tọa độ góc trên-trái của ảnh tìm thấy
            top_left_x, top_left_y = max_loc

            # Tính tọa độ giữa và convert về Python int để tránh lỗi JSON serialize
            center_x = int(top_left_x + template_width // 2)
            center_y = int(top_left_y + template_height // 2)

            return (center_x, center_y)
        else:
            return (None, None)

    except Exception as e:
        return (None, None)


def find_all_img_centers(
    d: "u2.Device",
    template_path: str,
    threshold: float = 0.85,
) -> list:
    """
    Tìm tất cả hình ảnh mẫu trên màn hình và trả về danh sách tọa độ giữa của các ảnh được tìm thấy.

    Args:
        d: Đối tượng thiết bị uiautomator2.
        template_path: Đường dẫn tương đối đến tệp hình ảnh mẫu trong thư mục "assets".
        threshold: Ngưỡng tương đồng để xác định một kết quả khớp (từ 0.0 đến 1.0).

    Returns:
        List các tuple (x, y) là tọa độ giữa của các ảnh được tìm thấy.
        Trả về list rỗng [] nếu không tìm thấy ảnh nào.
    """
    try:
        # 1. Chụp ảnh màn hình từ thiết bị
        img_rgb = d.screenshot(format="opencv")
        if img_rgb is None:
            print("Lỗi: Không thể chụp ảnh màn hình.")
            return []

        # 2. Đọc ảnh mẫu
        template = cv2.imread(f"assets/{template_path}")
        if template is None:
            print(f"Lỗi: Không thể đọc tệp mẫu tại: assets/{template_path}")
            return []

        # 3. Lấy kích thước của template
        template_height, template_width = template.shape[:2]

        # 4. Thực hiện so khớp mẫu
        res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)

        # 5. Tìm tất cả các vị trí có độ tương đồng >= threshold
        loc = np.where(res >= threshold)

        # 6. Tạo danh sách các tọa độ giữa
        centers = []

        # Lặp qua tất cả các điểm tìm thấy
        for pt in zip(*loc[::-1]):  # [::-1] để chuyển từ (y,x) sang (x,y)
            top_left_x, top_left_y = pt

            # Tính tọa độ giữa và convert về Python int để tránh lỗi JSON serialize
            center_x = int(top_left_x + template_width // 2)
            center_y = int(top_left_y + template_height // 2)

            centers.append((center_x, center_y))

        # 7. Loại bỏ các điểm trùng lặp (nếu có)
        # Sử dụng khoảng cách tối thiểu để loại bỏ các điểm quá gần nhau
        min_distance = max(template_width, template_height) // 2
        filtered_centers = []

        for center in centers:
            is_duplicate = False
            for existing_center in filtered_centers:
                distance = (
                    (center[0] - existing_center[0]) ** 2
                    + (center[1] - existing_center[1]) ** 2
                ) ** 0.5
                if distance < min_distance:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered_centers.append(center)

        return filtered_centers

    except Exception as e:
        print(f"Lỗi trong find_all_img_centers: {str(e)}")
        return []


def connect_to_emulator(device_id: str = "emulator-5556"):
    """
    Kết nối tới emulator-5556 sử dụng uiautomator2
    Returns: device object nếu kết nối thành công, None nếu thất bại
    """
    try:
        # Kết nối tới emulator với địa chỉ emulator-5556
        device = u2.connect(device_id)

        # Kiểm tra xem thiết bị có sẵn sàng không
        print("Đang kiểm tra kết nối...")
        device_info = device.info

        print(f"Kết nối thành công!")
        print(f"Device: {device_info.get('productName', 'Unknown')}")
        print(f"Version: {device_info.get('version', 'Unknown')}")
        print(f"SDK: {device_info.get('sdk', 'Unknown')}")
        print("=========================")
        return device

    except Exception as e:
        print(f"Lỗi kết nối: {str(e)}")
        print("Hãy đảm bảo:")
        print("1. Emulator đang chạy")
        print("2. ADB đã được cài đặt và trong PATH")
        print("3. USB debugging đã được bật")
        return None


def custom_sleep(base_seconds: float):
    """Custom sleep function with configurable delay adjustment"""
    total_sleep = base_seconds + 0
    if total_sleep > 0:
        time.sleep(total_sleep)


def wait_and_click(
    device: u2.Device,
    template_path: str,
    timeout: float = 30.0,
    check_interval: float = 0.5,
    threshold: float = 0.85,
    roi: tuple = None,
    click_offset: tuple = (0, 0),
) -> bool:
    """
    Đợi cho đến khi thấy ảnh mẫu xuất hiện trên màn hình, sau đó click vào nó.

    Args:
        device: Đối tượng thiết bị uiautomator2
        template_path: Đường dẫn tới file ảnh mẫu trong thư mục assets
        timeout: Thời gian tối đa để đợi (giây)
        check_interval: Khoảng thời gian giữa các lần kiểm tra (giây)
        threshold: Ngưỡng độ tương đồng để xác định khớp
        roi: Vùng tìm kiếm (x1, y1, x2, y2), None = toàn màn hình
        click_offset: Offset từ tâm ảnh để click (x_offset, y_offset)

    Returns:
        True nếu tìm thấy và click thành công, False nếu timeout

    Example:
        # Sử dụng cơ bản - đợi và click vào ảnh
        wait_and_click(device, "next_gieo.png")

        # Với timeout 10 giây
        wait_and_click(device, "next_gieo.png", timeout=10)

        # Với interval kiểm tra 0.2 giây (kiểm tra nhanh hơn)
        wait_and_click(device, "next_gieo.png", check_interval=0.2)

        # Click với offset (click 10 pixel bên phải, 5 pixel xuống dưới so với tâm)
        wait_and_click(device, "button.png", click_offset=(10, 5))
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        # Tìm vị trí ảnh
        x, y = find_img_center(device, template_path, threshold)

        if x is not None and y is not None:
            # Áp dụng offset nếu có
            click_x = x + click_offset[0]
            click_y = y + click_offset[1]

            # Click vào vị trí
            device.click(click_x, click_y)
            print(
                f"Đã tìm thấy và click vào {template_path} tại ({click_x}, {click_y})"
            )
            return True

        # Đợi trước khi kiểm tra lại
        time.sleep(check_interval)

    print(f"Timeout: Không tìm thấy {template_path} trong {timeout} giây")
    return False
