import tkinter as tk
import tkinter.font as tkFont
from tkinter import filedialog
from tkinter import messagebox
import webbrowser
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
import numpy as np
import open3d as o3d

class App:
    def __init__(self, root):
        #setting title
        root.title("Image to 3D model")
        #setting window size
        width=354
        height=204
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)

        GButton_831=tk.Button(root)
        GButton_831["bg"] = "#000000"
        ft = tkFont.Font(family='Times',size=10)
        GButton_831["font"] = ft
        GButton_831["fg"] = "#01aaed"
        GButton_831["justify"] = "center"
        GButton_831["text"] = "Donate me :)"
        GButton_831.place(x=280,y=10,width=70,height=40)
        GButton_831["command"] = self.GButton_831_command

        GButton_489=tk.Button(root)
        GButton_489["bg"] = "#000000"
        ft = tkFont.Font(family='Times',size=38)
        GButton_489["font"] = ft
        GButton_489["fg"] = "#01aaed"
        GButton_489["justify"] = "center"
        GButton_489["text"] = "Convert!"
        GButton_489.place(x=70,y=70,width=212,height=94)
        GButton_489["command"] = self.GButton_489_command

        GButton_280=tk.Button(root)
        GButton_280["bg"] = "#000000"
        ft = tkFont.Font(family='Times',size=10)
        GButton_280["font"] = ft
        GButton_280["fg"] = "#01aaed"
        GButton_280["justify"] = "center"
        GButton_280["text"] = "Info"
        GButton_280.place(x=20,y=10,width=60,height=40)
        GButton_280["command"] = self.GButton_280_command

    def GButton_831_command(self):
        donateweb = messagebox.askokcancel("Donate","Go to donate page?")
        if donateweb == True:
            webbrowser.open("https://www.buymeacoffee.com/myusuf")
        else: 
            messagebox.showinfo(":(","You did not donate :( Feel free to use this program anyways!")


    def GButton_489_command(self):
        try:
            ################Yapay Zekaya göndermeye hazırlık#################
            feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
            model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

            filename = filedialog.askopenfilename(title="Select an image to Convert")
            image = Image.open(filename)
            new_height = 480 if image.height > 480 else image.height
            new_height -= (new_height % 32)
            new_width = int(new_height * image.width / image.height)
            diff = new_width % 32
            new_width = new_width - diff if diff < 16 else new_width + 32 - diff
            new_size = (new_width, new_height)
            image = image.resize(new_size)

            inputs = feature_extractor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth

            pad = 16
            output = predicted_depth.squeeze().cpu().numpy() * 1000.0
            output = output[pad:-pad, pad:-pad]
            image = image.crop((pad, pad, image.width - pad, image.height - pad))

            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(image)
            ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax[1].imshow(output, cmap='plasma')
            ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

            ###############Yapay zekaya göndermek###################

            width, height = image.size

            depth_image = (output * 255 / np.max(output)).astype('uint8')
            image = np.array(image)

            depth_o3d = o3d.geometry.Image(depth_image)
            image_o3d = o3d.geometry.Image(image)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False)

            camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
            camera_intrinsic.set_intrinsics(width, height, 500, 500, width/2, height/2)

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

            ################Çıkışı alıp, kaydedip ekranda göstermek##################

            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=20.0)
            pcd = pcd.select_by_index(ind)

            pcd.estimate_normals()
            pcd.orient_normals_to_align_with_direction()

            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10, n_threads=1)[0]

            rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
            mesh.rotate(rotation, center=(0, 0, 0))

            o3d.io.write_triangle_mesh(f'./output.obj', mesh)

            o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
        except ValueError:
            messagebox.showerror("Error","There may a problem in this image, try another image if possible. If you want to use this image, try convert its format.")


    def GButton_280_command(self):
        messagebox.showinfo("Info","Version 1 - Program Developed by M. Yusuf - Open3D Generates the 3D model on web - This program is for using Open3D simple and create 3D models from images easily. - You can contact the developer here: m.yusuf.yildirim33@gmail.com, i will answer anything! - I will be happy if you donate me! You can donate me by clickling Donate Me button on main page of program!")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
