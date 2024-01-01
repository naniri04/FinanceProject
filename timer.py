import time

st_time = time.time()

def start_timer():
    global st_time
    st_time = time.time()

def stop_timer():
    print(f"⬜⬜⬜⬜ [ {round(time.time() - st_time, 6)} s ] ⬜⬜⬜⬜")