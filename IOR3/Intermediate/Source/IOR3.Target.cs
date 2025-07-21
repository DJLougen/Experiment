using UnrealBuildTool;

public class IOR3Target : TargetRules
{
	public IOR3Target(TargetInfo Target) : base(Target)
	{
		DefaultBuildSettings = BuildSettingsVersion.Latest;
		IncludeOrderVersion = EngineIncludeOrderVersion.Latest;
		Type = TargetType.Game;
		ExtraModuleNames.Add("IOR3");
	}
}
