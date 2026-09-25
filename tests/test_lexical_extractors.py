from framenet_extractor import extract_candidate_frames
from verbnet_extractor import extract_candidate_classes


def test_framenet_example():
    frames = extract_candidate_frames("put the cup on the table")
    assert isinstance(frames, list)
    assert frames
    assert any(frame["name"].lower() == "placing" for frame in frames)
    assert all("frame_id" in frame for frame in frames)
    assert all("name" in frame for frame in frames)
    assert all("definition" in frame for frame in frames)


def test_verbnet_example():
    classes = extract_candidate_classes("put the cup on the table")
    assert isinstance(classes, list)
    assert classes
    assert any(class_info["class_id"].startswith("put") for class_info in classes)
    assert all("class_id" in class_info for class_info in classes)
    assert all("thematic_roles" in class_info for class_info in classes)
    assert all("syntax_frames" in class_info for class_info in classes)
