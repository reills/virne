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
  id 814
  arrival_time 16828.63231492225
  lifetime 139.97016077289697
  num_nodes 11
  type "path"
  node [
    id 0
    label "0"
    cpu 43
    gpu 33
    rom 41
  ]
  node [
    id 1
    label "1"
    cpu 34
    gpu 27
    rom 40
  ]
  node [
    id 2
    label "2"
    cpu 48
    gpu 1
    rom 5
  ]
  node [
    id 3
    label "3"
    cpu 2
    gpu 26
    rom 8
  ]
  node [
    id 4
    label "4"
    cpu 46
    gpu 9
    rom 39
  ]
  node [
    id 5
    label "5"
    cpu 14
    gpu 9
    rom 4
  ]
  node [
    id 6
    label "6"
    cpu 23
    gpu 40
    rom 44
  ]
  node [
    id 7
    label "7"
    cpu 17
    gpu 24
    rom 48
  ]
  node [
    id 8
    label "8"
    cpu 14
    gpu 14
    rom 26
  ]
  node [
    id 9
    label "9"
    cpu 4
    gpu 11
    rom 32
  ]
  node [
    id 10
    label "10"
    cpu 5
    gpu 12
    rom 4
  ]
  edge [
    source 0
    target 1
    bw 44
  ]
  edge [
    source 1
    target 2
    bw 23
  ]
  edge [
    source 2
    target 3
    bw 42
  ]
  edge [
    source 3
    target 4
    bw 25
  ]
  edge [
    source 4
    target 5
    bw 40
  ]
  edge [
    source 5
    target 6
    bw 40
  ]
  edge [
    source 6
    target 7
    bw 1
  ]
  edge [
    source 7
    target 8
    bw 24
  ]
  edge [
    source 8
    target 9
    bw 45
  ]
  edge [
    source 9
    target 10
    bw 30
  ]
]
