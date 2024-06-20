import subprocess
import time




def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = process.communicate()
    return stdout.decode(), stderr.decode(), process.returncode

detector = "Textsnake"
recognizer = "ABINet"

image = "demo/additions/text_recog/img-40-preprocesed.png"

# --show --save_vis  --print-result


def infer(img_pth):
    
    command = f'python tools/infer.py {img_pth} --det {detector} --rec {recognizer} --save_vis'
    stdout, stderr, returncode = run_command(command)
    if returncode == 0:
        print("Command executed successfully!")
        print("Output:")
        print(stdout)
        
        return stdout
    else:
        print("Error executing command:")
        print(stderr)
        return stderr
    
        
        
if __name__ == "__main__":
    t0 = time.perf_counter()
    infer(image)
    t1 = time.perf_counter()
    total = (t1-t0)
    total_ms = ((t1-t0)*10**3)
    print(image)
    print(f"perf counter time taken = {total}s")
    print(f"perf counter time taken = {total_ms}ms")
