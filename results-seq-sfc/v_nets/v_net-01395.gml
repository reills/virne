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
  id 1395
  arrival_time 29361.427178824408
  lifetime 98.54760202651201
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 21
    gpu 26
    rom 4
  ]
  node [
    id 1
    label "1"
    cpu 9
    gpu 46
    rom 49
  ]
  node [
    id 2
    label "2"
    cpu 16
    gpu 5
    rom 45
  ]
  node [
    id 3
    label "3"
    cpu 11
    gpu 6
    rom 16
  ]
  node [
    id 4
    label "4"
    cpu 35
    gpu 7
    rom 40
  ]
  node [
    id 5
    label "5"
    cpu 1
    gpu 2
    rom 41
  ]
  node [
    id 6
    label "6"
    cpu 42
    gpu 0
    rom 26
  ]
  node [
    id 7
    label "7"
    cpu 30
    gpu 23
    rom 31
  ]
  node [
    id 8
    label "8"
    cpu 50
    gpu 49
    rom 30
  ]
  node [
    id 9
    label "9"
    cpu 0
    gpu 20
    rom 3
  ]
  edge [
    source 0
    target 1
    bw 27
  ]
  edge [
    source 1
    target 2
    bw 23
  ]
  edge [
    source 2
    target 3
    bw 33
  ]
  edge [
    source 3
    target 4
    bw 23
  ]
  edge [
    source 4
    target 5
    bw 17
  ]
  edge [
    source 5
    target 6
    bw 29
  ]
  edge [
    source 6
    target 7
    bw 5
  ]
  edge [
    source 7
    target 8
    bw 5
  ]
  edge [
    source 8
    target 9
    bw 30
  ]
]
