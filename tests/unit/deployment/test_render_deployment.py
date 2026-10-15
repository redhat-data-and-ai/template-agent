"""The deployment renderer: values.yaml is the source of truth.

The point of the renderer is that nobody edits a resource limit in two
places. The test that matters most is the drift check — if someone edits
`generated.yaml` by hand, CI has to say so.
"""

from __future__ import annotations

import pytest

from scripts.render_deployment import (
    COMPONENT_PATHS,
    REPO_ROOT,
    for_target,
    plan,
    render_components,
    render_patches,
    rewrite_components,
)

import yaml

VALUES = yaml.safe_load(
    (REPO_ROOT / "config" / "agent" / "deployment" / "values.yaml").read_text()
)


def _docs(text: str) -> dict[str, dict]:
    return {
        f"{d['kind']}/{d['metadata']['name']}": d for d in yaml.safe_load_all(text) if d
    }


class TestCommittedOutputIsCurrent:
    def test_no_overlay_has_drifted(self):
        """What `make render-deployment-check` enforces in CI."""
        stale = [
            path.relative_to(REPO_ROOT)
            for path, text in plan(VALUES).items()
            if not path.is_file() or path.read_text() != text
        ]
        assert not stale, (
            f"overlays are out of date with values.yaml: {stale}. "
            "Run `make render-deployment`."
        )


class TestValuesReachTheManifests:
    @pytest.fixture(scope="class")
    def openshift(self):
        return _docs(render_patches(for_target(VALUES, "openshift"), "openshift"))

    def test_replicas_come_from_values(self, openshift):
        assert (
            openshift["Deployment/agent"]["spec"]["replicas"]
            == VALUES["app"]["replicas"]
        )

    def test_agent_resources_come_from_values(self, openshift):
        container = openshift["Deployment/agent"]["spec"]["template"]["spec"][
            "containers"
        ][0]
        assert container["resources"]["limits"]["memory"] == str(
            VALUES["resources"]["limits"]["memory"]
        )

    def test_container_is_named_so_strategic_merge_edits_it(self, openshift):
        """Without the name, merge appends a second container."""
        container = openshift["Deployment/agent"]["spec"]["template"]["spec"][
            "containers"
        ][0]
        assert container["name"] == "agent"

    def test_hpa_bounds_come_from_values(self, openshift):
        spec = openshift["HorizontalPodAutoscaler/agent"]["spec"]
        assert spec["maxReplicas"] == VALUES["autoscaling"]["maxReplicas"]
        targets = [
            m["resource"]["target"]["averageUtilization"] for m in spec["metrics"]
        ]
        assert targets == [
            VALUES["autoscaling"]["targetCPUUtilizationPercentage"],
            VALUES["autoscaling"]["targetMemoryUtilizationPercentage"],
        ]


class TestDisabledThingsAreOmitted:
    """A patch against a resource the overlay does not have is a hard error."""

    def test_autoscaling_off_emits_no_hpa_patch(self):
        values = {**VALUES, "autoscaling": {"enabled": False}}
        assert "HorizontalPodAutoscaler/agent" not in _docs(
            render_patches(values, "openshift")
        )

    def test_route_off_emits_no_route_patch(self):
        values = {
            **VALUES,
            "networking": {**VALUES["networking"], "route": {"enabled": False}},
        }
        assert "Route/agent" not in _docs(render_patches(values, "openshift"))

    def test_kind_gets_no_openshift_only_resources(self):
        """Kind has no BuildConfig, Route or HPA to patch."""
        docs = _docs(render_patches(for_target(VALUES, "kind"), "kind"))
        assert not {"BuildConfig/agent", "Route/agent"} & set(docs)


class TestComponentToggles:
    def test_enabled_components_are_listed(self):
        block = render_components({"components": {"postgres": True, "redis": False}})
        assert COMPONENT_PATHS["postgres"] in block
        assert f"- {COMPONENT_PATHS['redis']}" not in block

    def test_disabled_components_say_why_rather_than_vanishing(self):
        block = render_components({"components": {"redis": False}})
        assert "redis: disabled in values.yaml" in block

    def test_rewriting_preserves_the_rest_of_the_file(self):
        """Comments elsewhere in a kustomization explain real decisions."""
        original = (
            "resources:\n  - ../../base  # keep me\n\n"
            "components:\n  - ../../components/old\n\n"
            "labels:\n  - pairs:\n      app: agent\n"
        )
        out = rewrite_components(original, "components:\n  - ../../components/new\n\n")
        assert "# keep me" in out
        assert "app: agent" in out
        assert "components/old" not in out
        assert "components/new" in out

    def test_a_file_with_no_components_block_is_an_error(self):
        with pytest.raises(SystemExit):
            rewrite_components("resources:\n  - ../../base\n", "components:\n\n")


class TestPerTargetOverrides:
    def test_kind_drops_the_eval_runner(self):
        """It mounts a PVC no local cluster creates."""
        assert for_target(VALUES, "kind")["components"]["evalRunner"] is False
        assert VALUES["components"]["evalRunner"] is True

    def test_overrides_are_deep_merged_not_replaced(self):
        kind = for_target(VALUES, "kind")
        # `app.replicas` is overridden; its siblings must survive.
        assert kind["app"]["name"] == VALUES["app"]["name"]

    def test_the_targets_key_never_reaches_a_manifest(self):
        assert "targets" not in for_target(VALUES, "openshift")


class TestScaleToZeroWins:
    """Patches apply after components, so the renderer must stand aside.

    Both of these were caught by the existing scale-to-zero suite the
    moment the renderer started emitting values the component had just
    removed or set.
    """

    @pytest.fixture
    def values(self):
        v = yaml.safe_load(yaml.safe_dump(VALUES))
        v["components"]["scaleToZero"] = True
        v["autoscaling"]["enabled"] = False
        return v

    def test_no_replica_count_is_emitted(self, values):
        """It would overwrite the 0 the component sets."""
        deployment = _docs(render_patches(values, "openshift"))["Deployment/agent"]
        assert "replicas" not in deployment["spec"]

    def test_no_hpa_patch_is_emitted_even_if_autoscaling_is_left_on(self, values):
        """The component deletes the HPA; patching it fails the render."""
        values["autoscaling"]["enabled"] = True
        assert "HorizontalPodAutoscaler/agent" not in _docs(
            render_patches(values, "openshift")
        )
