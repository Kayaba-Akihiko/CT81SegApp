#  Copyright (c) 2025. by Yi GU <gu.yi.gu4@naist.ac.jp>,
#  Biomedical Imaging Intelligence Laboratory,
#  Nara Institute of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.


from pathlib import Path
import json
import copy

import numpy as np
import numpy.typing as npt
import vtk
import imageio.v3 as iio

from xmodules.xutils import os_utils, metaimage_utils, vtk_utils


def main():
    this_file = Path(__file__)
    this_dir = this_file.parent
    output_dir = this_dir / f'out_{this_file.stem}'
    output_dir.mkdir(exist_ok=True, parents=True)

    labelmap_path = this_dir / 'pred_label.mha'

    config_path = this_dir / 'naist_totalsegmentator_81.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    labelmap, spacing, _ = metaimage_utils.read(labelmap_path)

    preview_labelmap(
        labelmap, spacing, config, output_dir
    )
    return
    vtk_labelmap = vtk_utils.np_image_to_vtk(
        labelmap, spacing, name='labelmap')

    ren, win = vtk_utils.new_renderer_window(
        window_size=(600, 1500),
    )

    color = vtk.vtkColorTransferFunction()
    scalar_opacity = vtk.vtkPiecewiseFunction()
    for _, (class_id, r, g, b, a) in config['color'].items():
        color.AddRGBPoint(class_id, r, g, b)
        scalar_opacity.AddPoint(class_id, a)

    print(config['shade'], type(config['shade']))
    volume = vtk_utils.build_volume(
        image=vtk_labelmap,
        color=color,
        scalar_opacity=scalar_opacity,
        interpolation='nearest',
        specular=config['specular'],
        specular_power=config['specular_power'],
        ambient=config['ambient'],
        diffuse=config['diffuse'],
        shade=config['shade'],
        blend_mode_to_composite=True,
        device='cpu',
    )

    camera_offset = 2400
    ren.Clear()
    ren.RemoveAllViewProps()
    ren.AddVolume(volume)

    res = vtk_utils.render_view_as_np_image(
        ren=ren,
        win=win,
        view='front',
        view_camera_position=vtk_labelmap.GetCenter(),
        view_camera_offset=camera_offset,
    )
    ren.Clear()
    ren.RemoveAllViewProps()

    iio.imwrite(output_dir / 'pred_label.png', res)



def preview_labelmap(
        labelmap,
        spacing,
        rendering_config: dict,
        output_dir: Path,
        device: str = 'cpu',
):
    import vtk
    from vtkmodules.util import numpy_support as vtknp


    colors = vtk.vtkNamedColors()
    colors.SetColor("BkgColor", [255, 255, 255, 255])

    # Create the renderer, the render window, and the interactor. The renderer
    # draws into the render window, the interactor enables mouse- and
    # keyboard-based interaction with the scene.
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.SetOffScreenRendering(1)
    renWin.AddRenderer(ren)
    # iren = vtk.vtkRenderWindowInteractor()
    # iren.SetRenderWindow(renWin)

    # The gradient opacity function is used to decrease the opacity
    # in the "flat" regions of the volume while maintaining the opacity
    # at the boundaries between tissue types.  The gradient is measured
    # as the amount by which the intensity changes over unit distance.
    # For most medical data, the unit distance is 1mm.
    # gradient_opacity = vtk.vtkPiecewiseFunction()

    # The color transfer function maps voxel intensities to colors.
    color = vtk.vtkColorTransferFunction()
    scalar_opacity = vtk.vtkPiecewiseFunction()

    prop = vtk.vtkVolumeProperty()

    rendering_config = copy.deepcopy(rendering_config)
    if 'color' in rendering_config:
        for class_name, (class_id, r, g, b, a) in rendering_config['color'].items():
            color.AddRGBPoint(class_id, r, g, b)
            scalar_opacity.AddPoint(class_id, a)
        del rendering_config['color']

    # The opacity transfer function is used to control the opacity
    # of different tissue types.
    # opacityValue = 0.4
    # scalar_opacity.AddPoint(0, 0, 0.5, 0.0)
    # scalar_opacity.AddPoint(1, opacityValue, 0.5, 0.0)
    # scalar_opacity.AddPoint(len(lut) - 1, opacityValue, 0.5, 0.0)
    # scalar_opacity.AddPoint(39, 0, 0.5, 0.0)
    # scalar_opacity.AddPoint(40, 0, 0.5, 0.0)

    # gradient_opacity.AddPoint(0, 1)
    # gradient_opacity.AddPoint(90, 1)
    # gradient_opacity.AddPoint(180, 1)
    prop.SetInterpolationTypeToNearest()
    if (specular := rendering_config.pop('specular', None)) is not None:
        prop.SetSpecular(specular)
    if (specular_power := rendering_config.pop('specular_power', None)) is not None:
        prop.SetSpecularPower(specular_power)
    if (ambient := rendering_config.pop('ambient', None)) is not None:
        prop.SetAmbient(ambient)
    if (diffuse := rendering_config.pop('diffuse', None)) is not None:
        prop.SetDiffuse(diffuse)
    if (scalar_opacity_unit_distance := rendering_config.pop('scalar_opacity_unit_distance', None)) is not None:
        prop.SetScalarOpacityUnitDistance(scalar_opacity_unit_distance)
    prop.SetColor(color)
    prop.SetScalarOpacity(scalar_opacity)
    # prop.SetGradientOpacity(gradient_opacity)

    prop.ShadeOff()
    if (shade := rendering_config.pop('shade', None)) is not None:
        if isinstance(shade, bool):
            if shade:
                prop.ShadeOn()
        elif isinstance(shade, str):
            shade_l = shade.lower()
            if shade_l in ['t', 'true', 'yes', 'on']:
                prop.ShadeOn()
            elif shade_l in ['f', 'false', 'no', 'off']:
                pass
            else:
                raise ValueError(f'Invalid shade value: {shade}')
        elif isinstance(shade, int):
            if shade == 1:
                prop.ShadeOn()
            elif shade == 0:
                pass
            else:
                raise ValueError(f'Invalid shade value: {shade}')
        else:
            raise ValueError(f'Invalid shade value: {shade}')

    # Set a background color for the renderer
    ren.SetBackground(colors.GetColor3d("BkgColor"))

    # Increase the size of the render window
    # renWin.SetSize(600, 600)
    renWin.SetSize(600, 1500)

    # renWin.Render()
    renWin.SetOffScreenRendering(True)

    # The following reader is used to read a series of 2D slices (images)
    # that compose the volume. The slice dimensions are set, and the
    # pixel spacing. The data Endianness must also be specified. The reader
    # uses the FilePrefix in combination with the slice number to construct
    # filenames using the format FilePrefix.%d. (In this case the FilePrefix
    # is the root name of the file: quarter.)
    # reader = vtk.vtkMetaImageReader()  # for .mhd or .mha
    # reader = vtk.vtkNIFTIImageReader()  # for .nii or .nii.gz
    # reader = vtk.vtkNrrdReader()  # for .nii or .nii.gz
    # reader.SetFileName(str(label_path))

    # labelmap: np.ndarray with shape (N, H, W)
    # spacing: np.ndarray with shape (3,)  (e.g., [dz, dy, dx])
    N, H, W = labelmap.shape

    # Create vtkImageData
    image = vtk.vtkImageData()
    image.SetDimensions(W, H, N)  # VTK expects (x, y, z) = (W, H, N)
    # Adjust this depending on your spacing order
    # If spacing = (dz, dy, dx):
    image.SetSpacing(float(spacing[2]), float(spacing[1]), float(spacing[0]))
    # If spacing is already (dx, dy, dz), then:
    # image.SetSpacing(*[float(s) for s in spacing])

    # Convert NumPy array to vtkArray
    # Ensure the dtype matches what you want (e.g., uint8, uint16, etc.)
    labelmap_np = np.ascontiguousarray(labelmap)  # shape (N, H, W), C-order
    vtk_array = vtknp.numpy_to_vtk(
        num_array=labelmap_np.ravel(order="C"),
        deep=True,
        array_type=vtk.VTK_UNSIGNED_SHORT,  # or VTK_UNSIGNED_CHAR / VTK_SHORT etc.
    )
    vtk_array.SetName("labelmap")
    image.GetPointData().SetScalars(vtk_array)

    if device == 'cpu':
        mapper = vtk.vtkFixedPointVolumeRayCastMapper()
    elif device == 'cuda':
        mapper = vtk.vtkGPUVolumeRayCastMapper()
    else:
        raise ValueError(f'Device {device} is not supported.')

    # mapper.SetInputConnection(reader.GetOutputPort())
    mapper.SetInputData(image)
    # mapper.UseDepthPassOn()
    # mapper.SetImageSampleDistance(0.05)
    # mapper.SetAutoAdjustSampleDistances(False)
    # print('vtkGPUVolumeRayCastMapper::GetSampleDistance: {}'.format(mapper.GetSampleDistance()))
    # print('vtkGPUVolumeRayCastMapper::GetUseDepthPass: {}'.format(mapper.GetUseDepthPass()))
    # print('vtkGPUVolumeRayCastMapper::GetScalarModeAsString: {}'.format(mapper.GetScalarModeAsString()))
    mapper.SetBlendModeToComposite()
    # mapper.SetBlendModeToAverageIntensity()

    # The vtkVolume is a vtkProp3D (like a vtkActor) and controls the position
    # and orientation of the volume in world coordinates.
    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(prop)

    # Finally, add the volume to the renderer
    ren.RemoveAllViewProps()
    ren.AddViewProp(volume)

    # Set up an initial view of the volume.  The focal point will be the
    # center of the volume, and the camera position will be 400mm to the
    # patient's left (which is our right).
    camera = ren.GetActiveCamera()
    c = volume.GetCenter()
    # print("volume.GetCenter(): ", c)
    camera_distance = 2400  # 700
    # camera_distance = 1200 #700
    camera.SetViewUp(0, 0, 1)
    camera.SetPosition(c[0], c[1] + camera_distance, c[2])
    camera.SetClippingRange(100, 5000)
    camera.SetFocalPoint(c[0], c[1], c[2])
    # camera.Azimuth(90.0)
    camera.Elevation(0.0)
    camera.Azimuth(180.0)

    # screenshot code:
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(renWin)
    w2if.Update()
    writer = vtk.vtkPNGWriter()
    save_path = output_dir / 'preview_front.png'
    writer.SetFileName(str(save_path))
    writer.SetInputData(w2if.GetOutput())
    writer.Write()
    ren.Clear()

    camera.Azimuth(-90.0)
    # screenshot code:
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(renWin)
    w2if.Update()
    save_path = output_dir / 'preview_right.png'
    writer.SetFileName(str(save_path))
    writer.SetInputData(w2if.GetOutput())
    writer.Write()
    ren.Clear()

    camera.Azimuth(-90.0)
    # screenshot code:
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(renWin)
    w2if.Update()
    save_path = output_dir / 'preview_back.png'
    writer.SetFileName(output_dir / 'preview_back.png')
    writer.SetInputData(w2if.GetOutput())
    writer.Write()
    ren.Clear()

    camera.Azimuth(-90.0)
    # screenshot code:
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(renWin)
    w2if.Update()
    save_path = output_dir / 'preview_left.png'
    writer.SetFileName(str(save_path))
    writer.SetInputData(w2if.GetOutput())
    writer.Write()
    ren.Clear()

if __name__ == '__main__':
    main()