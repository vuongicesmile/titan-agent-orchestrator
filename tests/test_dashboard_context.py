from titan_agent_orchestrator.orchestration.dashboard import dashboard_arguments, dashboard_route


def test_followup_preserves_building_and_updates_device():
    previous = {"filters": {"building": "Building A"}, "options": {"equipment": ["Main Water Meter A"]}}
    assert dashboard_arguments("Đổi thiết bị đo sang Main Water Meter A", previous) == {"building": "Building A", "equipment": "Main Water Meter A"}


def test_building_change_clears_incompatible_device():
    previous = {"filters": {"building": "Building A", "equipment": "Old meter"}, "options": {"building": ["Building A", "Building B"]}}
    assert dashboard_arguments("Đổi qua Building B", previous) == {"building": "Building B"}


def test_unknown_building_queries_empty_instead_of_reusing_previous():
    previous = {"filters": {"building": "Building A"}, "options": {"building": ["Building A"]}}
    assert dashboard_arguments("Đổi tòa nhà XYZ", previous)["building"] == "XYZ"


def test_partial_building_keyword_kept_as_search_term():
    assert dashboard_arguments("đổi tòa nhà: Retail") == {"building": "Retail"}
    assert dashboard_arguments("đổi tòa nhà: ha noi") == {"building": "ha noi"}


def test_time_comparison_keeps_report_workflow():
    assert dashboard_arguments("Tạo dashboard điện năng tiêu thụ tháng này") is None


def test_building_typo_routes_existing_overview():
    assert dashboard_arguments("tạo dashboard thống kê dữ liệu cho tòa builng A") == {"building": "Building A"}


def test_vietnamese_building_label_preserves_source_and_clears_device():
    previous = {"filters": {"building": "spo_sharepoint:BLD081", "equipment": "old", "unit": "kWh"},
                "options": {"building": ["spo_samples:BLD120", "spo_sharepoint:BLD120"]},
                "option_labels": {"building": {"spo_samples:BLD120": "Retail Hà Nội 120 · spo_samples:BLD120", "spo_sharepoint:BLD120": "Retail Hà Nội 120 · spo_sharepoint:BLD120"}}}
    assert dashboard_arguments("đổi tòa nhà : Retail Hà Nội 120", previous) == {"building": "spo_sharepoint:BLD120", "unit": "kWh"}


def test_vietnamese_building_change_without_dashboard_history():
    assert dashboard_arguments("đổi tòa nhà : Retail Hà Nội 120") == {"building": "Retail Hà Nội 120"}


def test_ambiguous_name_does_not_choose_arbitrary_source():
    previous = {"filters": {}, "option_labels": {"building": {"spo_samples:BLD120": "Retail Hà Nội 120 · spo_samples:BLD120", "spo_sharepoint:BLD120": "Retail Hà Nội 120 · spo_sharepoint:BLD120"}}}
    assert dashboard_arguments("đổi tòa nhà : Retail Hà Nội 120", previous) == {"building": "Retail Hà Nội 120"}


def test_energy_followup_stays_on_comparison_and_overview_is_explicit():
    previous = {"filters": {}, "options": {"building": ["Building A", "Building B"], "equipment": ["demo:MAIN-A", "demo:MAIN-B"]}}
    assert dashboard_route("đổi tòa nhà: Building A", "fm.get_energy_comparison", previous) == ("fm.get_energy_comparison", {"building": "Building A"})
    assert dashboard_route("đổi công tơ A", "fm.get_energy_comparison", previous) == ("fm.get_energy_comparison", {"equipment": "demo:MAIN-A"})
    assert dashboard_route("mở dashboard overview", "fm.get_energy_comparison", previous) == ("fm.get_bms_dashboard", {})


def test_comparison_does_not_inherit_inventory_filters():
    assert dashboard_route("So sánh điện năng tháng này với tháng trước", "fm.get_bms_dashboard", {"filters": {"building": "Retail"}}) == ("fm.get_energy_comparison", {"metric": "electricity"})


def test_water_comparison_and_followup_grouping():
    assert dashboard_route("So sánh nước tháng này với tháng trước theo thiết bị tại tòa nhà A", None, None) == ("fm.get_energy_comparison", {"metric": "water", "group_by": "equipment", "building": "Building A"})
    previous = {"filters": {"metric": "electricity", "building": "Building A", "equipment": "demo:MAIN-A", "object_type": "Energy"}}
    assert dashboard_route("đổi sang nước", "fm.get_energy_comparison", previous) == ("fm.get_energy_comparison", {"metric": "water", "building": "Building A"})
    assert dashboard_route("thông tin thiết bị MAIN-A là gì?", "fm.get_energy_comparison", previous) is None


def test_followup_without_dashboard_context_does_not_open_overview():
    assert dashboard_route("đổi tòa nhà: Building A", None, None) is None


def test_qa_context_inherits_only_explicit_reference():
    from titan_agent_orchestrator.orchestration.dashboard import question_with_dashboard_context
    previous = {'filters': {'building': 'Building B', 'equipment': 'demo:MAIN-B', 'token': 'not-allowed'}}
    question = question_with_dashboard_context('Tòa nhà đó có bao nhiêu thiết bị?', previous)
    assert 'Building B' in question
    assert 'demo:MAIN-B' not in question
    assert 'token' not in question
    assert 'demo:MAIN-B' in question_with_dashboard_context('Thông tin thiết bị đó?', previous)
    assert question_with_dashboard_context('Tòa nhà Building A có bao nhiêu thiết bị?', previous) == 'Tòa nhà Building A có bao nhiêu thiết bị?'
    assert question_with_dashboard_context('Tòa nhà đó?', None) == 'Tòa nhà đó?'
