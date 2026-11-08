add_rules("mode.debug", "mode.release")

add_repositories("my-repo repo")
add_requireconfs("**.lua", {version = "5.4.6", override = true})
add_requires("edopro-core", {optional = true})

add_requires(
    "ygopro-core 0.0.3", "pybind11 2.13.*", "fmt 10.2.*", "glog 0.6.0",
    "sqlite3 3.43.0+200", "concurrentqueue 1.0.4", "unordered_dense 4.4.*",
    "sqlitecpp 3.2.1", "xz 5.8.1")


target("ygopro0_ygoenv")
    add_rules("python.library")
    add_files("ygoenv/ygoenv/ygopro0/*.cpp")
    add_packages("pybind11", "fmt", "glog", "concurrentqueue", "sqlitecpp", "unordered_dense", "ygopro-core")
    set_languages("c++17")
    if is_mode("release") then
        set_policy("build.optimization.lto", true)
        add_cxxflags("-march=native")
    end
    add_includedirs("ygoenv")

    after_build(function (target)
        local install_target = "$(projectdir)/ygoenv/ygoenv/ygopro0"
        os.cp(target:targetfile(), install_target)
        print("Copy target to " .. install_target)
    end)


target("core_lua_probe")
    -- test helper: run a Lua file in a core duel (tools/check_category_sset.py)
    set_kind("binary")
    set_default(false)
    add_files("tests/core_lua_probe.cpp")
    add_packages("ygopro-core")
    set_languages("c++17")


target("duel_registry_stress")
    -- regression test: concurrent duel create/end (core registry race)
    set_kind("binary")
    set_default(false)
    add_files("tests/duel_registry_stress.cpp")
    add_packages("ygopro-core")
    add_includedirs("ygoenv")
    set_languages("c++17")


target("yrp_verify")
    -- test helper: replay a .yrp through the core like the reference client
    set_kind("binary")
    set_default(false)
    add_files("tests/yrp_verify.cpp")
    add_packages("ygopro-core", "xz", "sqlite3")
    set_languages("c++17")


target("ygopro_ygoenv")
    add_rules("python.library")
    add_files("ygoenv/ygoenv/ygopro/*.cpp")
    add_packages("pybind11", "fmt", "glog", "concurrentqueue", "sqlitecpp", "unordered_dense", "ygopro-core", "xz")
    set_languages("c++17")
    if is_mode("release") then
        set_policy("build.optimization.lto", true)
        add_cxxflags("-march=native")
    end
    add_includedirs("ygoenv")

    after_build(function (target)
        local install_target = "$(projectdir)/ygoenv/ygoenv/ygopro"
        os.cp(target:targetfile(), install_target)
        print("Copy target to " .. install_target)
    end)

target("edopro_ygoenv")
    add_rules("python.library")
    add_files("ygoenv/ygoenv/edopro/*.cpp")
    add_packages("pybind11", "fmt", "glog", "concurrentqueue", "sqlitecpp", "unordered_dense", "edopro-core")
    set_languages("c++17")
    if is_mode("release") then
        set_policy("build.optimization.lto", true)
        add_cxxflags("-march=native")
    end
    add_includedirs("ygoenv")

    after_build(function (target)
        local install_target = "$(projectdir)/ygoenv/ygoenv/edopro"
        os.cp(target:targetfile(), install_target)
        print("Copy target to " .. install_target)
    end)


target("alphazero_mcts")
    add_rules("python.library")
    add_files("mcts/mcts/alphazero/*.cpp")
    add_packages("pybind11")
    set_languages("c++17")
    if is_mode("release") then
        set_policy("build.optimization.lto", true)
        add_cxxflags("-march=native")
    end
    add_includedirs("mcts")

    after_build(function (target)
        local install_target = "$(projectdir)/mcts/mcts/alphazero"
        os.cp(target:targetfile(), install_target)
        print("Copy target to " .. install_target)
        os.run("pybind11-stubgen mcts.alphazero.alphazero_mcts -o %s", "$(projectdir)/mcts")
    end)
