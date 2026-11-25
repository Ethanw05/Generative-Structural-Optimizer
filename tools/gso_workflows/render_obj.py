"""Orthographic inspection of a real OBJ. Run with Blender in background mode."""
import argparse,sys
from pathlib import Path
import bpy
from mathutils import Vector,Matrix
p=argparse.ArgumentParser()
p.add_argument('mesh',type=Path);p.add_argument('image',type=Path)
p.add_argument('--target',nargs=3,type=float,required=True)
p.add_argument('--from-direction',nargs=3,type=float,required=True)
p.add_argument('--up',nargs=3,type=float,required=True)
p.add_argument('--width',type=float,required=True)
p.add_argument('--size',nargs=2,type=int,default=[1200,720])
a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
bpy.ops.wm.read_factory_settings(use_empty=True)
vertices=[];faces=[]
with a.mesh.open(encoding='utf8') as f:
    for row in f:
        cols=row.split()
        if not cols:continue
        if cols[0]=='v':vertices.append(tuple(float(n) for n in cols[1:4]))
        if cols[0]=='f':faces.append(tuple(int(n.split('/')[0])-1 for n in cols[1:]))
mesh=bpy.data.meshes.new(a.mesh.stem);mesh.from_pydata(vertices,[],faces);mesh.update()
scene=bpy.context.scene;body=bpy.data.objects.new(a.mesh.stem,mesh);scene.collection.objects.link(body)
body.color=(.24,.49,.65,1)
scene.render.engine='BLENDER_WORKBENCH';s=scene.display.shading;s.light='STUDIO';s.color_type='OBJECT';s.show_shadows=True;s.show_cavity=True;s.cavity_type='WORLD'
s.cavity_ridge_factor=.8;s.cavity_valley_factor=.9;s.background_type='WORLD'
world=bpy.data.worlds.new('Inspection');scene.world=world;world.color=(.86,.88,.90)
scene.render.resolution_x,scene.render.resolution_y=a.size;scene.render.resolution_percentage=100;scene.display.render_aa='16'
scene.view_settings.view_transform='Standard'
cam_data=bpy.data.cameras.new('Inspection');cam=bpy.data.objects.new('Inspection',cam_data);scene.collection.objects.link(cam)
back=Vector(a.from_direction).normalized();right=Vector(a.up).cross(back).normalized();up=back.cross(right).normalized()
cam.matrix_world=Matrix((right,up,back)).transposed().to_4x4()
cam.location=Vector(a.target)+back*(a.width*3);cam_data.type='ORTHO';cam_data.ortho_scale=a.width;cam_data.clip_end=max(1000,a.width*6)
scene.camera=cam;scene.render.filepath=str(a.image.resolve());bpy.ops.render.render(write_still=True)
