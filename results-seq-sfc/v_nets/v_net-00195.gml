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
  id 195
  arrival_time 3532.995084450944
  lifetime 2105.19011060762
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 8
    gpu 12
    rom 33
  ]
  node [
    id 1
    label "1"
    cpu 47
    gpu 16
    rom 17
  ]
  node [
    id 2
    label "2"
    cpu 44
    gpu 40
    rom 39
  ]
  node [
    id 3
    label "3"
    cpu 36
    gpu 34
    rom 16
  ]
  node [
    id 4
    label "4"
    cpu 1
    gpu 27
    rom 11
  ]
  node [
    id 5
    label "5"
    cpu 5
    gpu 31
    rom 12
  ]
  node [
    id 6
    label "6"
    cpu 15
    gpu 8
    rom 9
  ]
  node [
    id 7
    label "7"
    cpu 7
    gpu 7
    rom 16
  ]
  node [
    id 8
    label "8"
    cpu 24
    gpu 49
    rom 33
  ]
  node [
    id 9
    label "9"
    cpu 1
    gpu 32
    rom 50
  ]
  edge [
    source 0
    target 1
    bw 12
  ]
  edge [
    source 1
    target 2
    bw 46
  ]
  edge [
    source 2
    target 3
    bw 3
  ]
  edge [
    source 3
    target 4
    bw 35
  ]
  edge [
    source 4
    target 5
    bw 34
  ]
  edge [
    source 5
    target 6
    bw 12
  ]
  edge [
    source 6
    target 7
    bw 4
  ]
  edge [
    source 7
    target 8
    bw 25
  ]
  edge [
    source 8
    target 9
    bw 32
  ]
]
