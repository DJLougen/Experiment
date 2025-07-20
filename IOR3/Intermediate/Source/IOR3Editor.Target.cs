using UnrealBuildTool;

public class IOR3EditorTarget : TargetRules
{
	public IOR3EditorTarget(TargetInfo Target) : base(Target)
	{
		DefaultBuildSettings = BuildSettingsVersion.Latest;
		IncludeOrderVersion = EngineIncludeOrderVersion.Latest;
		Type = TargetType.Editor;
		ExtraModuleNames.Add("IOR3");
	}
}
