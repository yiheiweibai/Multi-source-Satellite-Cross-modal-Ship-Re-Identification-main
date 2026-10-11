"""上传 SDF-Net 代码 + 准备脚本 + 元数据到 AutoDL。"""
import stat
import paramiko
from pathlib import Path

HOST = "connect.cqa1.seetacloud.com"
PORT = 37128
USER = "root"
PASS = "Dh3DnQzqschK"

LOCAL_ROOT = Path(r"d:\Projects\Multi-source Satellite Cross-modal Ship Re-Identification")
SYS = LOCAL_ROOT / "ship_reid_system"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
sftp = ssh.open_sftp()


def mkdirs(path):
    parts = path.strip("/").split("/")
    cur = ""
    for p in parts:
        cur += "/" + p
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


SKIP_DIRS = {"__pycache__", "data", "logs", ".git", "figs", "weights", "sims"}
n_file = 0


def upload_dir(local: Path, remote: str):
    global n_file
    mkdirs(remote)
    for item in local.iterdir():
        if item.is_dir():
            if item.name in SKIP_DIRS:
                continue
            upload_dir(item, f"{remote}/{item.name}")
        elif item.suffix in (".py", ".yml", ".yaml", ".txt", ".md", ".json"):
            sftp.put(str(item), f"{remote}/{item.name}")
            n_file += 1
            if n_file % 10 == 0:
                print(f"  uploaded {n_file} files ...")


# 1. SDF-Net 代码
print("=== 上传 SDF-Net 代码 ===")
upload_dir(SYS / "SDF-Net", "/root/autodl-tmp/SDF-Net")
print(f"SDF-Net: {n_file} files")

# 2. scripts（prepare + integrate）
n0 = n_file
mkdirs("/root/autodl-tmp/scripts")
for s in ["sdfnet_prepare_data.py", "integrate_cyclegan.py", "autodl_sdfnet_finetune.sh"]:
    sftp.put(str(SYS / "scripts" / s), f"/root/autodl-tmp/scripts/{s}")
    n_file += 1
print(f"scripts: +{n_file - n0}")

# 3. 元数据
meta_local = LOCAL_ROOT / "赛题6-初赛" / "训练数据"
mkdirs("/root/autodl-tmp/meta")
for m in ["labels_train.csv", "labels.csv", "local_val_task.json"]:
    sftp.put(str(meta_local / m), f"/root/autodl-tmp/meta/{m}")
    print(f"meta: {m} ({(meta_local / m).stat().st_size} bytes)")

sftp.close()

# 4. 远程：为 integrate 脚本建期望的目录名（fake_sar->optical_to_sar）
print("\n=== 建立 aug 目录别名 ===")
stdin, stdout, stderr = ssh.exec_command(
    "cd /root/autodl-tmp/aug && "
    "mkdir -p HOSS_augmented && "
    "ln -sfn ../fake_sar HOSS_augmented/optical_to_sar && "
    "ln -sfn ../fake_optical HOSS_augmented/sar_to_optical && "
    "ls -la HOSS_augmented/", timeout=30)
print(stdout.read().decode())

# 磁盘空间
stdin, stdout, stderr = ssh.exec_command("df -h /root/autodl-tmp | tail -1", timeout=15)
print("磁盘:", stdout.read().decode().strip())

# 5. 远程一键部署 + 启动训练（pip/权重下载约 5-10 分钟）
print("\n=== 远程执行 autodl_sdfnet_finetune.sh ===")
stdin, stdout, stderr = ssh.exec_command(
    "bash /root/autodl-tmp/scripts/autodl_sdfnet_finetune.sh 2>&1", timeout=1800)
print(stdout.read().decode())
e = stderr.read().decode().strip()
if e:
    print("STDERR:", e[:1000])

ssh.close()
print("=== DONE ===")
