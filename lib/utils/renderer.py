import os
#os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['PYOPENGL_PLATFORM']  = 'osmesa'
import math
import trimesh
import pyrender
import numpy as np
from pyrender.constants import RenderFlags
from lib.models.smpl import get_smpl_faces


class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P


class Renderer:
    def __init__(self, resolution=(224,224), orig_img=False, wireframe=False):
        self.resolution = resolution

        self.faces = get_smpl_faces()
        self.orig_img = orig_img
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=1.0
        )

        # set the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        # light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=0.8)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)


    # 根据网格重建
    def render(self, img, verts, cam, angle=None, axis=None, mesh_filename=None, color=[1.0, 1.0, 0.9], rotate=False):
        # 使用了trimesh库创建了一个三维网格对象mesh，该网格对象由顶点（vertices）和面（faces）组成。
        # vertices是一个形状为(N, 3)的二维数组
        # faces的形状为(M, 3)，表示只有M个三角形面
        # process=False参数表示不对网格对象进行预处理，例如计算法向量、检查有效性等
        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)

        # 旋转矩阵Rx，用于绕X轴旋转180度
        Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        mesh.apply_transform(Rx)

        if rotate:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(60), [0, 1, 0])
            mesh.apply_transform(rot)

        if mesh_filename is not None:
            # 使用Trimesh库中的export方法将三角网格模型对象mesh保存为文件
            mesh.export(mesh_filename)

        if angle and axis:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh.apply_transform(R)

        sx, sy, tx, ty = cam

        camera = WeakPerspectiveCamera(
            scale=[sx, sy],
            translation=[tx, ty],
            zfar=1000.
        )

        # 使用pyrender库中的MetallicRoughnessMaterial类创建了一个材质对象material。该材质对象的各个属性如下：
        # metallicFactor: 金属系数，表示材质表面反射光的金属属性程度，取值范围为0到1，此处为0表示非金属材质。
        # alphaMode: 透明度模式，表示材质的透明度表现方式，此处为不透明模式。
        # smooth: 表示材质表面是否进行了光滑处理，此处为True表示进行了光滑处理。
        # wireframe: 是否显示网格线框架，此处为True表示显示。
        # roughnessFactor: 粗糙系数，表示材质表面的粗糙程度，取值范围为0到1，此处为1表示表面非常粗糙。
        # emissiveFactor: 发光系数，表示材质表面的发光强度，由一个三元组表示RGB颜色值，此处为(0.1, 0.1, 0.1)表示较低的发光强度。
        # baseColorFactor: 基础颜色系数，表示材质表面的颜色，由一个四元组表示RGBA颜色值，
        # 此处为(color[0], color[1], color[2], 1.0)，其中color是一个三元组，表示RGB颜色值，透明度为1.0表示不透明。
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            smooth=True,
            wireframe=True,
            roughnessFactor=1.0,
            emissiveFactor=(0.1, 0.1, 0.1),
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        # 将Trimesh对象转换为Pyrender中的Mesh对象，并设置了材质（material），
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        # 然后将Mesh对象添加到Pyrender中的场景（scene）中，并指定名称为mesh用于后续操作
        mesh_node = self.scene.add(mesh, 'mesh')

        # 使用pyrender库进行渲染。具体来说，它完成了以下几个步骤：
        # 创建相机节点：根据相机的参数（内参、外参），创建相机节点，并将其加入场景中。
        camera_pose = np.eye(4)
        # 创建渲染器：创建一个渲染器对象，用于将场景渲染成图像。
        cam_node = self.scene.add(camera, pose=camera_pose)

        # 设置渲染参数：根据要求的渲染效果，设置渲染参数，包括是否显示网格、是否进行背景透明等。
        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        # 渲染场景：调用渲染器的render函数，将场景渲染成图像。该函数的第一个参数是要渲染的场景，第二个参数是渲染参数。
        # 返回图像：将渲染得到的RGB图像和深度图像返回。
        rgb, _ = self.renderer.render(self.scene, flags=render_flags)
        # 生成一个有效掩码，即 valid_mask，用于过滤掉背景的无效像素
        valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
        output_img = rgb[:, :, :] * valid_mask + (1 - valid_mask) * img
        # 将输出图像转换为无符号 8 位整数类型，并将结果存储在变量 image 中
        image = output_img.astype(np.uint8)

        # 移除以便下一次渲染
        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)

        return image
