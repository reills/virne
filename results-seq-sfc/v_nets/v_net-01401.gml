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
  id 1401
  arrival_time 29415.84733502309
  lifetime 2713.161937638248
  num_nodes 11
  type "path"
  node [
    id 0
    label "0"
    cpu 17
    gpu 23
    rom 31
  ]
  node [
    id 1
    label "1"
    cpu 26
    gpu 13
    rom 8
  ]
  node [
    id 2
    label "2"
    cpu 23
    gpu 13
    rom 44
  ]
  node [
    id 3
    label "3"
    cpu 8
    gpu 5
    rom 6
  ]
  node [
    id 4
    label "4"
    cpu 1
    gpu 5
    rom 7
  ]
  node [
    id 5
    label "5"
    cpu 49
    gpu 42
    rom 39
  ]
  node [
    id 6
    label "6"
    cpu 23
    gpu 38
    rom 45
  ]
  node [
    id 7
    label "7"
    cpu 39
    gpu 4
    rom 4
  ]
  node [
    id 8
    label "8"
    cpu 29
    gpu 16
    rom 20
  ]
  node [
    id 9
    label "9"
    cpu 26
    gpu 28
    rom 50
  ]
  node [
    id 10
    label "10"
    cpu 20
    gpu 2
    rom 25
  ]
  edge [
    source 0
    target 1
    bw 15
  ]
  edge [
    source 1
    target 2
    bw 44
  ]
  edge [
    source 2
    target 3
    bw 8
  ]
  edge [
    source 3
    target 4
    bw 36
  ]
  edge [
    source 4
    target 5
    bw 1
  ]
  edge [
    source 5
    target 6
    bw 13
  ]
  edge [
    source 6
    target 7
    bw 41
  ]
  edge [
    source 7
    target 8
    bw 40
  ]
  edge [
    source 8
    target 9
    bw 49
  ]
  edge [
    source 9
    target 10
    bw 24
  ]
]
