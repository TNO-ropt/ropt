# Utility Functions

Optional helpers for scripts and applications that use `ropt`. Nothing in `ropt`
calls these. See [Plugin Discovery](../advanced/plugin_discovery.md) for the
plugin queries, [Variable scaling](../optimizer_setup/configuration_sections.md#variable-scales)
for the bounds converter, and [Keyboard
Interrupts](../troubleshooting/keyboard_interrupt.md) for when the escape hatch is
worth reaching for.

::: ropt.utils
    options:
        show_root_members_full_path: false
        members:
            - validate_backend_options
            - scales_and_offsets_from_bounds
            - restore_keyboard_interrupt
