import os
import shutil

def main(src_zarr):
    root_path = "/nrs/saalfeld/heinrichl/mlflow_tracking/diffusion/695944031355267611/55da5f83a58a4cb982a273c9687a7132/artifacts/checkpoints/ckpt_141"
    tgt_zarr = "samples.zarr"
    tgt_dirs = [dir_name for dir_name in os.listdir(os.path.join(root_path, tgt_zarr))if os.path.isdir(os.path.join(root_path, tgt_zarr, dir_name))]
    tgt_dirs = sorted(tgt_dirs, key=lambda x: int(x))
    digits = 6
    if len(tgt_dirs)> 0:
        highest = int(tgt_dirs[-1])
    else:
        highest = -1
    print(f"Starting with highest {highest}")
    dirs = [dir_name for dir_name in os.listdir(os.path.join(root_path, src_zarr))if os.path.isdir(os.path.join(root_path, src_zarr, dir_name))]
    dirs = sorted(dirs, key=lambda x: int(x))
    dirs = dirs[65001:]
    for subdir_name in dirs:
        print(subdir_name)
        new_ds = f"{highest + 1:06d}"
        print(f"move from {os.path.join(root_path, src_zarr, subdir_name)} to {os.path.join(root_path, tgt_zarr, new_ds)}")
        shutil.move(os.path.join(root_path, src_zarr, subdir_name), os.path.join(root_path, tgt_zarr, new_ds))
        highest = highest + 1
    #os.remove(os.path.join(root_path, src_zarr, ".zgroup"))
    #os.rmdir(os.path.join(root_path, src_zarr))
def other():
    root_path = "/nrs/saalfeld/heinrichl/mlflow_tracking/diffusion/695944031355267611/55da5f83a58a4cb982a273c9687a7132/artifacts/checkpoints/ckpt_141"
    tgt_zarr = "samples.zarr"
    dirs = os.listdir(os.path.join(root_path, tgt_zarr))
    dirs = [dir_name for dir_name in dirs if dir_name.startswith(" ")]
    print(dirs)
    target_dirs = [dir_name.replace(" ", "0") for dir_name in dirs]
    print(target_dirs)
    for src, tgt in zip(dirs, target_dirs):
        shutil.move(os.path.join(root_path, tgt_zarr, src), os.path.join(root_path, tgt_zarr, tgt))
    
def iterate_main():
    
# model_141.pt            samples_138365167.zarr  samples_138365185.zarr  samples_138365203.zarr  samples_138365221.zarr  samples_138367355.zarr  samples_138367651.zarr  samples_138374143.zarr  samples_138379982.zarr  samples_138380000.zarr  samples_138380019.zarr  samples_138380037.zarr  samples_138381195.zarr  samples_138381213.zarr  samples_138384523.zarr
# samples                 samples_138365168.zarr  samples_138365186.zarr  samples_138365204.zarr  samples_138365222.zarr  samples_138367356.zarr  samples_138367652.zarr  samples_138374144.zarr  samples_138379983.zarr  samples_138380001.zarr  samples_138380020.zarr  samples_138381178.zarr  samples_138381196.zarr  samples_138381214.zarr  samples_138384524.zarr
# samples_138365151.zarr  samples_138365169.zarr  samples_138365187.zarr  samples_138365205.zarr  samples_138365223.zarr  samples_138367622.zarr  samples_138367653.zarr  samples_138374145.zarr  samples_138379984.zarr  samples_138380002.zarr  samples_138380021.zarr  samples_138381179.zarr  samples_138381197.zarr  samples_138381215.zarr  samples_138384525.zarr
# samples_138365152.zarr  samples_138365170.zarr  samples_138365188.zarr  samples_138365206.zarr  samples_138365224.zarr  samples_138367623.zarr  samples_138374128.zarr  samples_138374146.zarr  samples_138379985.zarr  samples_138380003.zarr  samples_138380022.zarr  samples_138381180.zarr  samples_138381198.zarr  samples_138381216.zarr  samples_138384527.zarr
# samples_138365153.zarr  samples_138365171.zarr  samples_138365189.zarr  samples_138365207.zarr  samples_138365225.zarr  samples_138367624.zarr  samples_138374129.zarr  samples_138374147.zarr  samples_138379986.zarr  samples_138380004.zarr  samples_138380023.zarr  samples_138381181.zarr  samples_138381199.zarr  samples_138381217.zarr  samples_138384528.zarr
# samples_138365154.zarr  samples_138365172.zarr  samples_138365190.zarr  samples_138365208.zarr  samples_138365226.zarr  samples_138367625.zarr  samples_138374130.zarr  samples_138377098.zarr  samples_138379987.zarr  samples_138380005.zarr  samples_138380024.zarr  samples_138381182.zarr  samples_138381200.zarr  samples_138384479.zarr  samples_138384529.zarr
# samples_138365155.zarr  samples_138365173.zarr  samples_138365191.zarr  samples_138365209.zarr  samples_138365227.zarr  samples_138367626.zarr  samples_138374131.zarr  samples_138377100.zarr  samples_138379988.zarr  samples_138380006.zarr  samples_138380025.zarr  samples_138381183.zarr  samples_138381201.zarr  samples_138384480.zarr  samples_138384530.zarr
# samples_138365156.zarr  samples_138365174.zarr  samples_138365192.zarr  samples_138365210.zarr  samples_138365228.zarr  samples_138367627.zarr  samples_138374132.zarr  samples_138377102.zarr  samples_138379989.zarr  samples_138380007.zarr  samples_138380026.zarr  samples_138381184.zarr  samples_138381202.zarr  samples_138384481.zarr  samples_138384556.zarr
# samples_138365157.zarr  samples_138365175.zarr  samples_138365193.zarr  samples_138365211.zarr  samples_138365229.zarr  samples_138367628.zarr  samples_138374133.zarr  samples_138377103.zarr  samples_138379990.zarr  samples_138380008.zarr  samples_138380027.zarr  samples_138381185.zarr  samples_138381203.zarr  samples_138384482.zarr  samples_138384558.zarr
# samples_138365158.zarr  samples_138365176.zarr  samples_138365194.zarr  samples_138365212.zarr  samples_138365230.zarr  samples_138367629.zarr  samples_138374134.zarr  samples_138377104.zarr  samples_138379991.zarr  samples_138380009.zarr  samples_138380028.zarr  samples_138381186.zarr  samples_138381204.zarr  samples_138384483.zarr  samples_138384559.zarr
# samples_138365159.zarr  samples_138365177.zarr  samples_138365195.zarr  samples_138365213.zarr  samples_138367319.zarr  samples_138367630.zarr  samples_138374135.zarr  samples_138377105.zarr  samples_138379992.zarr  samples_138380010.zarr  samples_138380029.zarr  samples_138381187.zarr  samples_138381205.zarr  samples_138384485.zarr  samples_138384560.zarr
# samples_138365160.zarr  samples_138365178.zarr  samples_138365196.zarr  samples_138365214.zarr  samples_138367348.zarr  samples_138367644.zarr  samples_138374136.zarr  samples_138377106.zarr  samples_138379993.zarr  samples_138380011.zarr  samples_138380030.zarr  samples_138381188.zarr  samples_138381206.zarr  samples_138384486.zarr  samples_138384561.zarr
# samples_138365161.zarr  samples_138365179.zarr  samples_138365197.zarr  samples_138365215.zarr  samples_138367349.zarr  samples_138367645.zarr  samples_138374137.zarr  samples_138377107.zarr  samples_138379994.zarr  samples_138380012.zarr  samples_138380031.zarr  samples_138381189.zarr  samples_138381207.zarr  samples_138384487.zarr  samples_138384562.zarr
# samples_138365162.zarr  samples_138365180.zarr  samples_138365198.zarr  samples_138365216.zarr  samples_138367350.zarr  samples_138367646.zarr  samples_138374138.zarr  samples_138379977.zarr  samples_138379995.zarr  samples_138380013.zarr  samples_138380032.zarr  samples_138381190.zarr  samples_138381208.zarr  samples_138384488.zarr  samples_138384563.zarr
# samples_138365163.zarr  samples_138365181.zarr  samples_138365199.zarr  samples_138365217.zarr  samples_138367351.zarr  samples_138367647.zarr  samples_138374139.zarr  samples_138379978.zarr  samples_138379996.zarr  samples_138380014.zarr  samples_138380033.zarr  samples_138381191.zarr  samples_138381209.zarr  samples_138384489.zarr  samples.zarr
# samples_138365164.zarr  samples_138365182.zarr  samples_138365200.zarr  samples_138365218.zarr  samples_138367352.zarr  samples_138367648.zarr  samples_138374140.zarr  samples_138379979.zarr  samples_138379997.zarr  samples_138380015.zarr  samples_138380034.zarr  samples_138381192.zarr  samples_138381210.zarr  samples_138384520.zarr
# samples_138365165.zarr  samples_138365183.zarr  samples_138365201.zarr  samples_138365219.zarr  samples_138367353.zarr  samples_138367649.zarr  samples_138374141.zarr  samples_138379980.zarr  samples_138379998.zarr  samples_138380016.zarr  samples_138380035.zarr  samples_138381193.zarr  samples_138381211.zarr  samples_138384521.zarr
# samples_138365166.zarr  samples_138365184.zarr  samples_138365202.zarr  samples_138365220.zarr  samples_138367354.zarr  samples_138367650.zarr  samples_138374142.zarr  samples_138379981.zarr  samples_138379999.zarr  samples_138380018.zarr  samples_138380036.zarr  samples_138381194.zarr  samples_138381212.zarr  samples_138384522.zarr
    src_zarrs = [
"samples_138365151.zarr",
"samples_138365152.zarr",
"samples_138365153.zarr",
"samples_138365154.zarr",
"samples_138365155.zarr",
"samples_138365156.zarr",
"samples_138365157.zarr",
"samples_138365158.zarr",
"samples_138365159.zarr",
"samples_138365160.zarr",
"samples_138365161.zarr",
"samples_138365162.zarr",
"samples_138365163.zarr",
"samples_138365164.zarr",
"samples_138365165.zarr",
"samples_138365166.zarr",
"samples_138365167.zarr",
"samples_138365168.zarr",
"samples_138365169.zarr",
"samples_138365170.zarr",
"samples_138365171.zarr",
"samples_138365172.zarr",
"samples_138365173.zarr",
"samples_138365174.zarr",
"samples_138365175.zarr",
"samples_138365176.zarr",
"samples_138365177.zarr",
"samples_138365178.zarr",
"samples_138365179.zarr",
"samples_138365180.zarr",
"samples_138365181.zarr",
"samples_138365182.zarr",
"samples_138365183.zarr",
"samples_138365184.zarr",
"samples_138365185.zarr",
"samples_138365186.zarr",
"samples_138365187.zarr",
"samples_138365188.zarr",
"samples_138365189.zarr",
"samples_138365190.zarr",
"samples_138365191.zarr",
"samples_138365192.zarr",
"samples_138365193.zarr",
"samples_138365194.zarr",
"samples_138365195.zarr",
"samples_138365196.zarr",
"samples_138365197.zarr",
"samples_138365198.zarr",
"samples_138365199.zarr",
"samples_138365200.zarr",
"samples_138365201.zarr",
"samples_138365202.zarr",
"samples_138365203.zarr",
"samples_138365204.zarr",
"samples_138365205.zarr",
"samples_138365206.zarr",
"samples_138365207.zarr",
"samples_138365208.zarr",
"samples_138365209.zarr",
"samples_138365210.zarr",
"samples_138365211.zarr",
"samples_138365212.zarr",
"samples_138365213.zarr",
"samples_138365214.zarr",
"samples_138365215.zarr",
"samples_138365216.zarr",
"samples_138365217.zarr",
"samples_138365218.zarr",
"samples_138365219.zarr",
"samples_138365220.zarr",
    ]
    for src_zarr in src_zarrs:
        print(f"Running on {src_zarr}")
        main(src_zarr)
if __name__ == "__main__":
    main("samples_second.zarr")
    