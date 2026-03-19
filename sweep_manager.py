import subprocess
import os
import time
import requests
from datetime import datetime

# --- 配置区 ---
SWEEP_PATH = os.environ.get("SWEEP_CONFIG_PATH", "sweep_config/2026-01-12-Elliptic-train-rate.yaml") # 你的 Sweep ID
GPUS = os.environ.get("GPUS", "0")            # 想要运行的 GPU 列表
LOG_ROOT = os.environ.get("LOG_ROOT", "logs")                 # 日志存放目录
WEBHOOK_URL = os.environ['WANDB_NOTIFY_URL']   # 用于通知的 Webhook 地址

GPU_LIST = [g.strip() for g in GPUS.split(",") if g.strip()]

# 用于接收第一次返回的 tg message id
tg_message_id = None
# 更新间隔（秒）
UPDATE_INTERVAL = 15


def send_notification(content):  
    print(f"发送通知: {content}")  
    payload = {"text": content, "timestamp": str(datetime.now())} 
    try:  
        requests.post(WEBHOOK_URL, json=payload, timeout=10)  
    except Exception as e:  
        print(f"发送通知失败: {e}")  

def send_sweep_info(sweep_fullname, status):
    global tg_message_id  
    print(f"Sweep Notification to {WEBHOOK_URL} (Status: {status})")
    
    payload = {
        "sweep_id": sweep_fullname.split('/')[-1], 
        "project": sweep_fullname.split('/')[1], 
        "entity": sweep_fullname.split('/')[0], 
        "status": status, 
        "timestamp": str(datetime.now())
    }
    
    # 只要不是第一次启动，且我们已经拿到了消息 ID，就带上它用于修改消息
    if status in ["running", "end"] and tg_message_id:
        payload["message_id"] = tg_message_id

    try:  
        response = requests.post(WEBHOOK_URL, json=payload, timeout=10)  
        
        # 第一次启动时获取 ID
        if status == "start":
            res_data = response.json()
            tg_message_id = res_data.get("message_id")
            print(f"成功获取并保存 Telegram Message ID: {tg_message_id}")
            
    except Exception as e:  
        print(f"发送通知失败: {e}")


def run_agents():  
    # 1. 创建 Sweep 并获取 ID
    print(f"[{datetime.now()}] 正在创建 Sweep...")
    cmd_create = ["wandb", "sweep", SWEEP_PATH]
    result = subprocess.run(cmd_create, capture_output=True, text=True)
    
    sweep_id = ""
    for line in (result.stderr + result.stdout).split('\n'):
        if "wandb agent" in line:
            sweep_fullname = line.split("wandb agent ")[-1].strip()
            sweep_id = sweep_fullname.split('/')[-1]
            break
            
    if not sweep_id:
        print("无法创建 Sweep，请检查配置文件")
        return

    print(f"Sweep 创建成功: {sweep_id}")
    # send_notification(f"🚀 Sweep 已启动\nSweep ID: {sweep_id}\nGPUs: {GPUS}\nSweep 网址: https://wandb.ai/HCCS/VoxGFormer/sweeps/{sweep_id}")
    send_sweep_info(sweep_fullname=sweep_fullname, status="start")
    start_time = datetime.now()
    # 2. 启动 Agents
    processes = []
    # 使用解析后的 GPU_LIST 进行循环
    for gpu_id in GPU_LIST:  
        # 为每个 GPU 创建独立的日志文件
        log_file_path = os.path.join(LOG_ROOT, f"agent_gpu_{gpu_id}.log")
        log_file = open(log_file_path, "w")  
        
        env = os.environ.copy()  
        env["CUDA_VISIBLE_DEVICES"] = gpu_id  
          
        p = subprocess.Popen(  
            ["wandb", "agent", sweep_fullname],  
            env=env,  
            stdout=log_file,  
            stderr=subprocess.STDOUT,  
            text=True  
        )  
        processes.append((gpu_id, p, log_file))  
        print(f"GPU {gpu_id} 上的 Agent 已启动 (PID: {p.pid})")  

    last_update_time = time.time()
    # 2. 轮询检查进程状态
    try:
        while True:
            alive_processes = [p for _, p, _ in processes if p.poll() is None]
            if not alive_processes:
                break
            
            # --- 新增：定期自动发送 RUNNING 通知 ---
            current_time = time.time()

            if current_time - last_update_time > UPDATE_INTERVAL:
                print(f"[{datetime.now()}] 执行定时进度刷新...")
                send_sweep_info(sweep_fullname=sweep_fullname, status="running")
                last_update_time = current_time # 重置计时器
            
            time.sleep(10) 
    except KeyboardInterrupt:
        print("\n正在停止所有 Agent...")
        for _, p, _ in processes:
            p.terminate()

    # 3. 扫尾工作
    end_time = datetime.now()
    duration = end_time - start_time
    
    # 关闭所有日志文件
    for _, _, log_file in processes:
        log_file.close()

    send_sweep_info(sweep_fullname=sweep_fullname, status="end")

if __name__ == "__main__":
    run_agents()