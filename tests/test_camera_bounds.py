"""Collision extents, partial-frustum intersections, and surface distances."""
from math import pi, sqrt
from types import SimpleNamespace as NS

import pytest

from test_search_execution_grounding import node, supervisor, world_utils
from external.webots_ros2_simulation.controllers.fallback_action_supervisor.camera_grounding import (
    object_camera_measurement, select_camera_candidate, visible_bounds_surface,
    DISTANCE_AMBIGUITY_TOLERANCE_M,
)

IDENTITY = [1,0,0,0,1,0,0,0,1]
GEOMETRY = ([0,0,0],IDENTITY,pi/2,pi/2)
ALIGNMENT = ([0,0,0],IDENTITY)


def box(identifier, position, size, kind='Table', rotation=IDENTITY):
    result = node(identifier, f'object_{identifier}', position=position)
    old_fields = result.getField.side_effect
    primitive = NS(getTypeName=lambda:'Box',
                   getField=lambda name:NS(getSFVec3f=lambda:size))
    result.getField.side_effect = lambda name: (NS(getSFNode=lambda:primitive)
                                               if name == 'boundingObject' else old_fields(name))
    result.getTypeName.return_value = kind
    result.getOrientation.return_value = rotation
    return result


def measurement(obj):
    return object_camera_measurement(obj,GEOMETRY,ALIGNMENT)


def test_large_object_origin_outside_but_edge_visible():
    obj = box(1,(3,0,2),(3,1,1),'RoundTable')
    result = measurement(obj)
    assert not result['origin_visible']
    assert result['visible']
    assert result['distance'] == pytest.approx(sqrt(1.5**2 + 1.5**2))
    # Verified RoundTable subtype reaches the normal execution resolver.
    result = world_utils.resolve_execution_instance(supervisor(obj),'table_1','table',
        use_camera=True,camera_context={'camera':GEOMETRY,'scene_to_map':ALIGNMENT})
    assert result['target'] == 'object_1'
    assert result['optical_target'] == pytest.approx([1.5,0,1.5])
    assert any('origin_visible=no bounding_volume_visible=yes' in line
               for line in result['camera_diagnostics'])


@pytest.mark.parametrize('position', [(5,0,2),(0,5,2),(0,0,-2)])
def test_entire_bounds_outside_or_behind_rejected(position):
    obj = box(1,position,(1,1,1))
    assert not measurement(obj)['visible']
    with pytest.raises(ValueError,match='no visible compatible candidate'):
        select_camera_candidate([('object_1',obj)],GEOMETRY)


def test_small_centered_object_uses_front_surface():
    result = measurement(box(1,(0,0,2),(.2,.2,.2)))
    assert result['origin_visible'] and result['visible']
    assert result['nearest_visible_point'] == pytest.approx([0,0,1.9])


def test_face_intersects_frustum_even_when_no_corners_are_inside():
    # Camera frustum pierces the middle of a large face. Corner sampling fails.
    result = measurement(box(1,(0,0,3),(20,20,1)))
    assert result['visible']
    assert result['distance'] == pytest.approx(2.5)


def test_nearest_surface_beats_nearest_origin():
    small = box(1,(0,0,3),(.2,.2,.2))
    large = box(2,(0,0,5),(1,1,6))
    selected,_ = select_camera_candidate([('small',small),('large',large)],GEOMETRY)
    assert selected[0][0] == 'large'


def test_nearly_equal_surface_distances_are_ambiguous():
    assert DISTANCE_AMBIGUITY_TOLERANCE_M == .05
    a = box(1,(0,0,3),(1,1,1))
    b = box(2,(0,0,4),(1,1,2.96))
    with pytest.raises(ValueError,match='ambiguous'):
        select_camera_candidate([('a',a),('b',b)],GEOMETRY)


def test_object_rotation_and_scene_alignment_are_applied_once():
    obj = box(1,(3,0,2),(1,4,1),rotation=[0,-1,0,1,0,0,0,0,1])
    result = measurement(obj)
    assert result['visible']
    assert result['distance'] == pytest.approx(sqrt(1+1.5**2))
    point = visible_bounds_surface(([-2,-.5,-.5],[2,.5,.5]),[3,0,2],
                                   ([10,20,0],IDENTITY,pi/2,pi/2),([10,20,0],IDENTITY))
    assert point == pytest.approx(result['nearest_visible_point'])


def test_missing_extent_does_not_silently_use_origin():
    obj = box(1,(0,0,2),(1,1,1))
    obj.getField.side_effect = lambda name:None
    obj.getBaseNodeField.return_value = None
    result = measurement(obj)
    assert result['origin_visible']
    assert not result['bounds_available'] and not result['visible']


def test_compound_cylinder_bounds_come_from_fields_not_table_constants():
    obj = box(1,(0,0,2),(1,1,1))
    cylinder = NS(getTypeName=lambda:'Cylinder', getField=lambda name:
                  NS(getSFFloat=lambda:{'radius':.6,'height':.08}[name]))
    transform = NS(getTypeName=lambda:'Transform',getField=lambda name:{
        'translation':NS(getSFVec3f=lambda:[0,0,.7]),
        'children':NS(getTypeName=lambda:'MFNode',getCount=lambda:1,getMFNode=lambda i:cylinder),
    }.get(name))
    obj.getField.side_effect = lambda name: NS(getSFNode=lambda:transform) if name == 'boundingObject' else None
    result = measurement(obj)
    assert result['nearest_visible_point'] == pytest.approx([0,0,2.66])
