// BepInEx plugin for the Stationeers dedicated server: once game data is loaded, write the logic
// tables of every ILogicable prefab as JSON to $SICC_DUMP_OUT, then quit.
//
// The tables are computed the way Stationpedia.PopulateThingPages does (AddLogicTypeInfo,
// AddLogicSlotTypeInfo, AddSlotInfo, AddLogicModeInfo, AddInstructionInfo). The server builds
// Stationpedia pages too; the logic tables are cross-checked against them, and any mismatch fails
// the run. The tables are computed rather than read off the pages because hidden prefabs have no
// page.
//
// Text (names, descriptions, instruction formats) is written as the Stationpedia shows it: macros
// such as {THING:...} expanded by Localization.ParseHelpText, TextMeshPro rich-text tags kept.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;
using Assets.Scripts;
using Assets.Scripts.Localization2;
using Assets.Scripts.Objects;
using Assets.Scripts.Objects.Electrical;
using Assets.Scripts.Objects.Motherboards;
using Assets.Scripts.Objects.Pipes;
using Assets.Scripts.UI;
using BepInEx;
using Newtonsoft.Json;
using Objects.Rockets;
using UnityEngine;

[BepInPlugin("sicc.logicdump", "SiccLogicDump", "1.0.0")]
public class SiccLogicDump : BaseUnityPlugin
{
    // Messages go through UnityEngine.Debug (the server's -logFile): BepInEx's own disk log is not
    // reliably flushed before Application.Quit.
    void Awake()
    {
        WorldManager.OnGameDataLoaded += OnGameDataLoaded;
        Debug.Log("SiccLogicDump: waiting for WorldManager.OnGameDataLoaded");
    }

    static void OnGameDataLoaded()
    {
        int code = 0;
        try
        {
            string outPath = Environment.GetEnvironmentVariable("SICC_DUMP_OUT");
            if (string.IsNullOrEmpty(outPath))
                throw new Exception("SICC_DUMP_OUT is not set");
            List<Entry> entries = Collect();
            Write(outPath, entries);
            Debug.Log("SiccLogicDump: wrote " + entries.Count + " prefabs to " + outPath);
            if (!CrossCheck(entries))
                code = 3;
        }
        catch (Exception e)
        {
            Debug.LogError("SiccLogicDump: FAILED\n" + e);
            code = 1;
        }
        Debug.Log("SiccLogicDump: exiting with code " + code);
        Application.Quit(code);
    }

    class Entry
    {
        public Thing Prefab;
        public bool Hidden;
        public string Name;
        public string Description;
        public List<KeyValuePair<string, string>> Logic = new List<KeyValuePair<string, string>>();
        public List<KeyValuePair<string, List<int>>> SlotLogic = new List<KeyValuePair<string, List<int>>>();
        public List<KeyValuePair<string, List<int>>> SlotLogicWrite = new List<KeyValuePair<string, List<int>>>();
        public List<string> Modes = new List<string>();
        // null: no IMemory
        public Memory Memory;
    }

    class Memory
    {
        // 0 when the size depends on the instance (a circuit housing's chip)
        public int Size;
        public string Access;
        // the IInstructable enum name (a key of InstructionSets), or null
        public string Instructions;
    }

    class Instruction
    {
        public string Name;
        public int OpCode;
        public string Format;
    }

    // IInstructable enum name -> instructions; devices that share an enum describe it identically
    static readonly SortedDictionary<string, List<Instruction>> InstructionSets =
        new SortedDictionary<string, List<Instruction>>(StringComparer.Ordinal);

    // Structure.IsStructureCompleted gates Device.CanLogicWrite(LogicSlotType, int) even outside a
    // running game, and a prefab sits at its first build state.
    static readonly FieldInfo CurrentBuildState =
        typeof(Structure).GetField("_currentBuildState", BindingFlags.NonPublic | BindingFlags.Instance)
        ?? throw new Exception("Structure._currentBuildState not found");

    static List<Entry> Collect()
    {
        Stationpedia.DataHandler.HandleThingPageOverrides();
        Debug.Log("SiccLogicDump: prefabs=" + Prefab.AllPrefabs.Count
            + " HiddenInPedia=" + Stationpedia.DataHandler.HiddenInPedia.Count);

        var entries = new List<Entry>();
        foreach (Thing prefab in Prefab.AllPrefabs.OrderBy(p => p.PrefabName, StringComparer.Ordinal))
        {
            ILogicable logicable = prefab as ILogicable;
            if (logicable == null)
                continue;

            bool hiddenOverride;
            Stationpedia.DataHandler.HiddenInPedia.TryGetValue(prefab.PrefabName, out hiddenOverride);
            var e = new Entry
            {
                Prefab = prefab,
                Hidden = prefab.HideInStationpedia || hiddenOverride,
                Name = Localization.GetThingName(prefab.PrefabName),
                Description = Localization.ParseHelpText(Localization.GetThingDescription(prefab.PrefabName)),
            };

            DeviceInputOutput dio = logicable as DeviceInputOutput;
            if (dio != null)
            {
                foreach (Connection openEnd in dio.OpenEnds)
                    openEnd.Validate();
            }
            foreach (LogicType lt in EnumCollections.LogicTypes.Values)
            {
                bool r = logicable.CanLogicRead(lt);
                bool w = logicable.CanLogicWrite(lt);
                if (r || w)
                    e.Logic.Add(new KeyValuePair<string, string>(lt.ToString(), (r ? "R" : "") + (w ? "W" : "")));
            }

            if (prefab.Slots != null)
            {
                foreach (LogicSlotType st in Logicable.LogicSlotTypes)
                {
                    var idx = new List<int>();
                    for (int i = 0; i < prefab.Slots.Count; i++)
                    {
                        if (prefab.Slots[i] != null && logicable.CanLogicRead(st, i))
                            idx.Add(i);
                    }
                    if (idx.Count > 0)
                        e.SlotLogic.Add(new KeyValuePair<string, List<int>>(st.ToString(), idx));
                }
                ISlotWriteable slotWriteable = logicable as ISlotWriteable;
                Structure structure = prefab as Structure;
                if (slotWriteable != null)
                {
                    object buildState = structure != null ? CurrentBuildState.GetValue(structure) : null;
                    if (structure != null)
                        CurrentBuildState.SetValue(structure, structure.BuildStates.Count - 1);
                    try
                    {
                        foreach (LogicSlotType st in Logicable.LogicSlotTypes)
                        {
                            var idx = new List<int>();
                            for (int i = 0; i < prefab.Slots.Count; i++)
                            {
                                if (prefab.Slots[i] != null && slotWriteable.CanLogicWrite(st, i))
                                    idx.Add(i);
                            }
                            if (idx.Count > 0)
                                e.SlotLogicWrite.Add(new KeyValuePair<string, List<int>>(st.ToString(), idx));
                        }
                    }
                    finally
                    {
                        if (structure != null)
                            CurrentBuildState.SetValue(structure, buildState);
                    }
                }
            }

            IMemory memory = prefab as IMemory;
            if (memory != null)
                e.Memory = CollectMemory(prefab, memory);

            if ((prefab is RocketAvionicsDevice || prefab.HasModeState) && !(prefab is LogicDial) && prefab.ModeStrings != null)
                e.Modes.AddRange(prefab.ModeStrings);

            entries.Add(e);
        }
        return entries;
    }

    // as Stationpedia.AddInstructionInfo
    static Memory CollectMemory(Thing prefab, IMemory memory)
    {
        bool r = memory is IMemoryReadable;
        bool w = memory is IMemoryWritable;
        var m = new Memory { Size = memory.GetStackSize(), Access = (r ? "R" : "") + (w ? "W" : "") };
        IInstructable instructable = memory as IInstructable;
        if (instructable == null)
            return m;

        IEnumCollection collection = instructable.GetInstructions();
        m.Instructions = collection.GetEnumTypeName();
        var instructions = new List<Instruction>();
        // index 0 is None
        for (int i = 1; i < collection.Length; i++)
        {
            instructions.Add(new Instruction
            {
                Name = collection.GetNameFromIndex(i),
                OpCode = collection.GetIntFromIndex(i),
                Format = instructable.GetInstructionDescription(i),
            });
        }
        List<Instruction> known;
        if (!InstructionSets.TryGetValue(m.Instructions, out known))
            InstructionSets[m.Instructions] = instructions;
        else if (!instructions.Select(Describe).SequenceEqual(known.Select(Describe)))
            throw new Exception(prefab.PrefabName + " describes " + m.Instructions + " differently from another device");
        return m;
    }

    static string Describe(Instruction i)
    {
        return i.Name + "=" + i.OpCode + "\n" + i.Format;
    }

    static void Write(string outPath, List<Entry> entries)
    {
        using (var sw = new StreamWriter(outPath))
        using (var w = new JsonTextWriter(sw))
        {
            w.Formatting = Formatting.Indented;
            w.Indentation = 1;
            w.WriteStartObject();
            w.WritePropertyName("game_version");
            w.WriteValue(GameManager.GetGameVersion());

            w.WritePropertyName("enums");
            w.WriteStartObject();
            WriteEnum<LogicType>(w, "LogicType");
            WriteEnum<LogicSlotType>(w, "LogicSlotType");
            WriteEnum<ColorType>(w, "ColorType");
            w.WriteEndObject();

            w.WritePropertyName("docs");
            w.WriteStartObject();
            w.WritePropertyName("LogicType");
            w.WriteStartObject();
            foreach (LogicType lt in EnumCollections.LogicTypes.Values)
            {
                w.WritePropertyName(lt.ToString());
                w.WriteValue(Localization.ParseHelpText(LogicBase.GetLogicDescription(lt)));
            }
            w.WriteEndObject();
            w.WritePropertyName("LogicSlotType");
            w.WriteStartObject();
            foreach (LogicSlotType st in Logicable.LogicSlotTypes)
            {
                w.WritePropertyName(st.ToString());
                w.WriteValue(Localization.ParseHelpText(LogicBase.GetLogicDescription(st)));
            }
            w.WriteEndObject();
            w.WriteEndObject();

            w.WritePropertyName("instruction_sets");
            w.WriteStartObject();
            foreach (var kv in InstructionSets)
            {
                w.WritePropertyName(kv.Key);
                w.WriteStartArray();
                foreach (Instruction i in kv.Value)
                    w.WriteRawValue(JsonConvert.SerializeObject(new { name = i.Name, opcode = i.OpCode, format = i.Format }));
                w.WriteEndArray();
            }
            w.WriteEndObject();

            w.WritePropertyName("prefabs");
            w.WriteStartArray();
            foreach (Entry e in entries)
            {
                w.WriteStartObject();
                w.WritePropertyName("prefab_name");
                w.WriteValue(e.Prefab.PrefabName);
                w.WritePropertyName("prefab_hash");
                w.WriteValue(e.Prefab.PrefabHash);
                w.WritePropertyName("class");
                w.WriteValue(e.Prefab.GetType().FullName);
                w.WritePropertyName("hidden");
                w.WriteValue(e.Hidden);
                w.WritePropertyName("name");
                w.WriteValue(e.Name);
                w.WritePropertyName("description");
                w.WriteValue(e.Description);
                // CableNetwork.DataDeviceList (what batch instructions and pins address) is List<Device>
                w.WritePropertyName("device");
                w.WriteValue(e.Prefab is Device);

                w.WritePropertyName("logic");
                w.WriteStartObject();
                foreach (var kv in e.Logic)
                {
                    w.WritePropertyName(kv.Key);
                    w.WriteValue(kv.Value);
                }
                w.WriteEndObject();

                w.WritePropertyName("slot_logic");
                w.WriteStartObject();
                foreach (var kv in e.SlotLogic)
                {
                    w.WritePropertyName(kv.Key);
                    w.WriteRawValue("[" + string.Join(", ", kv.Value) + "]");
                }
                w.WriteEndObject();

                w.WritePropertyName("slot_logic_write");
                w.WriteStartObject();
                foreach (var kv in e.SlotLogicWrite)
                {
                    w.WritePropertyName(kv.Key);
                    w.WriteRawValue("[" + string.Join(", ", kv.Value) + "]");
                }
                w.WriteEndObject();

                w.WritePropertyName("slots");
                w.WriteStartArray();
                if (e.Prefab.HasAnySlots)
                {
                    foreach (Slot slot in e.Prefab.Slots)
                        w.WriteRawValue(JsonConvert.SerializeObject(new { name = slot.StringKey, type = slot.Type.ToString() }));
                }
                w.WriteEndArray();

                w.WritePropertyName("modes");
                w.WriteRawValue(JsonConvert.SerializeObject(e.Modes));

                w.WritePropertyName("memory");
                if (e.Memory == null)
                    w.WriteNull();
                else
                    w.WriteRawValue(JsonConvert.SerializeObject(new { size = e.Memory.Size, access = e.Memory.Access, instructions = e.Memory.Instructions }));

                w.WriteEndObject();
            }
            w.WriteEndArray();
            w.WriteEndObject();
            sw.WriteLine();
        }
    }

    static void WriteEnum<T>(JsonTextWriter w, string name) where T : struct
    {
        w.WritePropertyName(name);
        w.WriteStartObject();
        foreach (T v in (T[])Enum.GetValues(typeof(T)))
        {
            w.WritePropertyName(v.ToString());
            w.WriteValue(Convert.ToInt64(v));
        }
        w.WriteEndObject();
    }

    static readonly Regex LogicTypeLink = new Regex("link=LogicType(\\w+)>");
    static readonly Regex LogicSlotTypeLink = new Regex("link=LogicSlotType(\\w+)>");

    static string Access(string pediaAccess)
    {
        if (pediaAccess == GameStrings.LogicRead.DisplayString) return "R";
        if (pediaAccess == GameStrings.LogicWrite.DisplayString) return "W";
        if (pediaAccess == GameStrings.LogicReadWrite.DisplayString) return "RW";
        throw new Exception("unknown Stationpedia access string: " + pediaAccess);
    }

    static string Name(Regex re, string pediaName)
    {
        Match m = re.Match(pediaName);
        if (!m.Success)
            throw new Exception("cannot parse Stationpedia logic name: " + pediaName);
        return m.Groups[1].Value;
    }

    // Returns false on any mismatch between the computed tables and the game's own Stationpedia.
    static bool CrossCheck(List<Entry> entries)
    {
        var pages = new Dictionary<string, StationpediaPage>();
        foreach (StationpediaPage p in Stationpedia.StationpediaPages)
            pages[p.Key] = p;

        int compared = 0, missing = 0, mismatched = 0;
        foreach (Entry e in entries)
        {
            if (e.Hidden)
                continue;
            StationpediaPage page;
            if (!pages.TryGetValue("Thing" + e.Prefab.PrefabName, out page))
            {
                missing++;
                continue;
            }
            compared++;
            string mine = string.Join(" ", e.Logic.Select(kv => kv.Key + ":" + kv.Value))
                + " | " + string.Join(" ", e.SlotLogic.Select(kv => kv.Key + ":" + string.Join(", ", kv.Value)));
            string theirs = string.Join(" ", page.LogicInsert.Select(i => Name(LogicTypeLink, i.LogicName) + ":" + Access(i.LogicAccessTypes)))
                + " | " + string.Join(" ", page.LogicSlotInsert.Select(i => Name(LogicSlotTypeLink, i.LogicName) + ":" + i.LogicAccessTypes));
            if (mine != theirs)
            {
                mismatched++;
                if (mismatched <= 20)
                    Debug.LogError("SiccLogicDump: MISMATCH " + e.Prefab.PrefabName + "\n  computed:    " + mine + "\n  stationpedia: " + theirs);
            }
        }
        Debug.Log("SiccLogicDump: cross-check: Stationpedia.Instance=" + (Stationpedia.Instance != null)
            + " pages=" + pages.Count + " compared=" + compared + " no_page=" + missing + " mismatched=" + mismatched);
        if (compared == 0)
            Debug.LogError("SiccLogicDump: no Stationpedia pages to cross-check against");
        return compared > 0 && mismatched == 0;
    }
}
