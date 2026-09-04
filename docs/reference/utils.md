# Utility Functions

Optional helpers for scripts and applications that use `ropt`. Nothing in `ropt`
calls these. See [Plugin Discovery](../utilities/plugin_discovery.md) for the
plugin queries, [Variable scaling](../optimizer_setup/configuration.md#variable-scales)
for the bounds converter, and [Keyboard
Interrupts](../utilities/keyboard_interrupt.md) for when the escape hatch is
worth reaching for.

`ropt.utils` also re-exports
[`get_plugin_name`][ropt.plugins.manager.get_plugin_name], documented with the
[Plugin Manager](plugin_manager.md).

::: ropt.utils
    options:
        show_root_members_full_path: false
        members:
            - validate_backend_options
            - scales_and_offsets_from_bounds
            - restore_keyboard_interrupt
