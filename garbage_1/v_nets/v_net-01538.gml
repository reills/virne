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
  id 1538
  arrival_time 33984.32795162288
  lifetime 1358.9774449632573
  num_nodes 11
  type "path"
  node [
    id 0
    label "0"
    cpu 42
    gpu 8
    rom 5
  ]
  node [
    id 1
    label "1"
    cpu 22
    gpu 37
    rom 42
  ]
  node [
    id 2
    label "2"
    cpu 44
    gpu 12
    rom 10
  ]
  node [
    id 3
    label "3"
    cpu 12
    gpu 7
    rom 24
  ]
  node [
    id 4
    label "4"
    cpu 41
    gpu 38
    rom 10
  ]
  node [
    id 5
    label "5"
    cpu 23
    gpu 37
    rom 31
  ]
  node [
    id 6
    label "6"
    cpu 37
    gpu 47
    rom 12
  ]
  node [
    id 7
    label "7"
    cpu 33
    gpu 17
    rom 49
  ]
  node [
    id 8
    label "8"
    cpu 14
    gpu 45
    rom 21
  ]
  node [
    id 9
    label "9"
    cpu 34
    gpu 33
    rom 36
  ]
  node [
    id 10
    label "10"
    cpu 27
    gpu 24
    rom 8
  ]
  edge [
    source 0
    target 1
    bw 2
  ]
  edge [
    source 1
    target 2
    bw 31
  ]
  edge [
    source 2
    target 3
    bw 44
  ]
  edge [
    source 3
    target 4
    bw 3
  ]
  edge [
    source 4
    target 5
    bw 27
  ]
  edge [
    source 5
    target 6
    bw 47
  ]
  edge [
    source 6
    target 7
    bw 23
  ]
  edge [
    source 7
    target 8
    bw 20
  ]
  edge [
    source 8
    target 9
    bw 11
  ]
  edge [
    source 9
    target 10
    bw 9
  ]
]
