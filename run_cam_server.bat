@echo off

:: compile
aarch64-none-linux-gnu-gcc -O3 -funroll-loops -flto -o cam_server cam_server.c
if %errorlevel% neq 0 (
    echo Compile failed. Aborting.
    exit /b 1
)
echo Compile ok.
adb push cam_server /userdata
adb shell chmod +x /userdata/cam_server


:: exit low power mode
for /f %%i in ('adb shell "[ -f /sense/cfg/disable_ground_standby_mode ] && echo exists || echo not_exists"') do set FILE_STATUS=%%i

if "%FILE_STATUS%" == "exists" (
    echo The file /sense/cfg/disable_ground_standby_mode already exists. No action needed.
) else (
    echo The file /sense/cfg/disable_ground_standby_mode does not exist. Copying and rebooting...
    adb shell "mv /sense/cfg/enable_ground_standby_mode /sense/cfg/disable_ground_standby_mode"
    adb shell sync
    echo exit low power mode please reboot
    exit /b 1
)


adb shell touch /tmp/sv

adb shell rm /userdata/run_cam_server.sh
adb shell touch /userdata/run_cam_server.sh
adb shell "echo '/etc/init.d/usb3.0-gadget.sh restart rndis' >> /userdata/run_cam_server.sh"
adb shell "echo 'ifconfig usb0 192.168.0.101 netmask 255.255.255.0' >> /userdata/run_cam_server.sh"
adb shell "echo '/userdata/cam_server' >> /userdata/run_cam_server.sh"
adb shell chmod +x /userdata/run_cam_server.sh



adb shell "nohup sh /userdata/run_cam_server.sh 1>/userdata/output.log 2>&1 & sleep 1"
