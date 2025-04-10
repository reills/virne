graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "gpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "rom"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  link_attrs_setting "_networkx_list_start"
  link_attrs_setting [
    name "bw"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "link"
    type "resource"
  ]
  id 981
  arrival_time 20936.557098890673
  lifetime 2868.826010073701
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 29
    gpu 49
    rom 30
  ]
  node [
    id 1
    label "1"
    cpu 41
    gpu 29
    rom 41
  ]
  node [
    id 2
    label "2"
    cpu 19
    gpu 7
    rom 35
  ]
  node [
    id 3
    label "3"
    cpu 41
    gpu 50
    rom 22
  ]
  node [
    id 4
    label "4"
    cpu 22
    gpu 34
    rom 5
  ]
  node [
    id 5
    label "5"
    cpu 17
    gpu 15
    rom 8
  ]
  node [
    id 6
    label "6"
    cpu 10
    gpu 36
    rom 40
  ]
  node [
    id 7
    label "7"
    cpu 6
    gpu 19
    rom 31
  ]
  node [
    id 8
    label "8"
    cpu 4
    gpu 42
    rom 46
  ]
  node [
    id 9
    label "9"
    cpu 34
    gpu 2
    rom 19
  ]
  edge [
    source 0
    target 1
    bw 8
  ]
  edge [
    source 1
    target 2
    bw 49
  ]
  edge [
    source 2
    target 3
    bw 2
  ]
  edge [
    source 3
    target 4
    bw 21
  ]
  edge [
    source 4
    target 5
    bw 35
  ]
  edge [
    source 5
    target 6
    bw 6
  ]
  edge [
    source 6
    target 7
    bw 1
  ]
  edge [
    source 7
    target 8
    bw 23
  ]
  edge [
    source 8
    target 9
    bw 6
  ]
]
