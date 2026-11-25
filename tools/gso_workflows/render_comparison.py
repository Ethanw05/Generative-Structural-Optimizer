"""Blender background renderer for matched engineering meshes and per-boundary-face scalar fields."""
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix,Vector


def render(mesh_file, output, views, scale, palette):
    data=np.load(mesh_file)
    mesh=bpy.data.meshes.new('material boundary')
    mesh.from_pydata(data['v'].tolist(),[],data['f'].tolist())
    mesh.update()
    body=bpy.data.objects.new('part',mesh);scene.collection.objects.link(body)
    is_field='vm' in data
    if is_field:
        rgba=palette[np.rint(255*np.clip(data['vm']/scale,0,1)).astype(int)]
        color=mesh.color_attributes.new(name='von Mises', type='FLOAT_COLOR', domain='CORNER')
        color.data.foreach_set('color',np.repeat(rgba,3,axis=0).astype(np.float32).ravel())
        mesh.color_attributes.active_color=color
    body.color=(.29,.47,.58,1.)
    shade=scene.display.shading
    shade.color_type='VERTEX' if is_field else 'OBJECT'
    shade.light='FLAT' if is_field else 'STUDIO'
    shade.show_cavity=not is_field;shade.cavity_type='WORLD';shade.cavity_ridge_factor=.7;shade.cavity_valley_factor=.85
    shade.show_shadows=not is_field;shade.show_specular_highlight=not is_field
    for i,(target,back,right,width) in enumerate(views):
        back=Vector(back).normalized();right=Vector(right).normalized();up=back.cross(right).normalized()
        cam.matrix_world=Matrix((right,up,back)).transposed().to_4x4()
        cam.location=Vector(target)+back*(width*3)
        cam.data.ortho_scale=width;cam.data.clip_end=2000.
        scene.render.filepath=str(output.parent/(output.name+f'-{i}.png'))
        bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(body,do_unlink=True);bpy.data.meshes.remove(mesh)


if __name__=='__main__':
    args=sys.argv[sys.argv.index('--')+1:]
    cfg=np.load(args[0])
    views=list(zip(cfg['target'],cfg['back'],cfg['right'],cfg['width']))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene;scene.render.engine='BLENDER_WORKBENCH'
    scene.view_settings.view_transform='Standard';scene.display.render_aa='16'
    scene.render.film_transparent=True
    scene.render.resolution_x=1000;scene.render.resolution_y=360;scene.render.resolution_percentage=100
    cam=bpy.data.objects.new('comparison camera',bpy.data.cameras.new('orthographic'));scene.collection.objects.link(cam)
    cam.data.type='ORTHO';scene.camera=cam
    for j in range(1,len(args),2):render(args[j],Path(args[j+1]),views,float(cfg['scale']),cfg['palette'])
