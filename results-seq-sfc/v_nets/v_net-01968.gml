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
  id 1968
  arrival_time 42978.88466778225
  lifetime 412.5665870464902
  num_nodes 11
  type "path"
  node [
    id 0
    label "0"
    cpu 7
    gpu 49
    rom 46
  ]
  node [
    id 1
    label "1"
    cpu 0
    gpu 8
    rom 49
  ]
  node [
    id 2
    label "2"
    cpu 29
    gpu 43
    rom 3
  ]
  node [
    id 3
    label "3"
    cpu 6
    gpu 46
    rom 23
  ]
  node [
    id 4
    label "4"
    cpu 30
    gpu 39
    rom 3
  ]
  node [
    id 5
    label "5"
    cpu 39
    gpu 45
    rom 30
  ]
  node [
    id 6
    label "6"
    cpu 20
    gpu 3
    rom 42
  ]
  node [
    id 7
    label "7"
    cpu 32
    gpu 41
    rom 45
  ]
  node [
    id 8
    label "8"
    cpu 9
    gpu 8
    rom 48
  ]
  node [
    id 9
    label "9"
    cpu 6
    gpu 5
    rom 8
  ]
  node [
    id 10
    label "10"
    cpu 21
    gpu 8
    rom 12
  ]
  edge [
    source 0
    target 1
    bw 32
  ]
  edge [
    source 1
    target 2
    bw 50
  ]
  edge [
    source 2
    target 3
    bw 0
  ]
  edge [
    source 3
    target 4
    bw 20
  ]
  edge [
    source 4
    target 5
    bw 42
  ]
  edge [
    source 5
    target 6
    bw 50
  ]
  edge [
    source 6
    target 7
    bw 49
  ]
  edge [
    source 7
    target 8
    bw 43
  ]
  edge [
    source 8
    target 9
    bw 7
  ]
  edge [
    source 9
    target 10
    bw 34
  ]
]
